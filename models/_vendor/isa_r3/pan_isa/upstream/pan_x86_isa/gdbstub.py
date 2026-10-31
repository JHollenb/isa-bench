"""Minimal GDB remote serial protocol support for the PAN x86 runtime."""
from __future__ import annotations

import socket
from typing import Optional

from .memory import MemoryFault
from .runtime import GDB_REG_ORDER, PanRuntime, SIGNAL_ILL, SIGNAL_SEGV, SIGNAL_TRAP


SEGMENT_REG_ORDER = ("CS", "SS", "DS", "ES", "FS", "GS")


REGISTER_XML = """<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target>
  <architecture>i386:x86-64</architecture>
  <feature name="org.gnu.gdb.i386.core">
    <reg name="rax" bitsize="64" type="int64"/>
    <reg name="rbx" bitsize="64" type="int64"/>
    <reg name="rcx" bitsize="64" type="int64"/>
    <reg name="rdx" bitsize="64" type="int64"/>
    <reg name="rsi" bitsize="64" type="int64"/>
    <reg name="rdi" bitsize="64" type="int64"/>
    <reg name="rbp" bitsize="64" type="data_ptr"/>
    <reg name="rsp" bitsize="64" type="data_ptr"/>
    <reg name="r8" bitsize="64" type="int64"/>
    <reg name="r9" bitsize="64" type="int64"/>
    <reg name="r10" bitsize="64" type="int64"/>
    <reg name="r11" bitsize="64" type="int64"/>
    <reg name="r12" bitsize="64" type="int64"/>
    <reg name="r13" bitsize="64" type="int64"/>
    <reg name="r14" bitsize="64" type="int64"/>
    <reg name="r15" bitsize="64" type="int64"/>
    <reg name="rip" bitsize="64" type="code_ptr"/>
    <reg name="eflags" bitsize="32" type="uint32"/>
  </feature>
</target>
"""


def checksum(payload: str) -> str:
    return f"{sum(payload.encode('ascii')) & 0xFF:02x}"


def frame(payload: str) -> bytes:
    return f"${payload}#{checksum(payload)}".encode("ascii")


def _hex_le(value: int, size: int) -> str:
    return int(value).to_bytes(size, "little").hex()


def _parse_hex_le(text: str) -> int:
    return int.from_bytes(bytes.fromhex(text), "little")


class GdbRemoteProtocol:
    def __init__(self, runtime: PanRuntime):
        self.runtime = runtime
        self.no_ack = False

    def stop_reply(self) -> str:
        reason = self.runtime.stop_reason
        if reason is None:
            return f"S{SIGNAL_TRAP:02x}"
        if reason.kind == "exit":
            return f"W{int(reason.exit_code or 0):02x}"
        signal = reason.signal or SIGNAL_TRAP
        return f"T{signal:02x}thread:1;"

    def read_registers(self) -> str:
        parts = [_hex_le(self.runtime.state.gpr[name], 8) for name in GDB_REG_ORDER]
        parts.append(_hex_le(self.runtime.state.rflags & 0xFFFFFFFF, 4))
        return "".join(parts)

    def read_register(self, index: int) -> str:
        if index < len(GDB_REG_ORDER):
            return _hex_le(self.runtime.state.gpr[GDB_REG_ORDER[index]], 8)
        if index == len(GDB_REG_ORDER):
            return _hex_le(self.runtime.state.rflags & 0xFFFFFFFF, 4)
        segment_index = index - len(GDB_REG_ORDER) - 1
        if 0 <= segment_index < len(SEGMENT_REG_ORDER):
            return _hex_le(self.runtime.state.segments.get(SEGMENT_REG_ORDER[segment_index], 0), 4)
        return "E01"

    def write_register(self, index: int, data_hex: str) -> str:
        value = _parse_hex_le(data_hex)
        if index < len(GDB_REG_ORDER):
            self.runtime.state.gpr[GDB_REG_ORDER[index]] = value
            return "OK"
        if index == len(GDB_REG_ORDER):
            self.runtime.state.rflags = value
            return "OK"
        segment_index = index - len(GDB_REG_ORDER) - 1
        if 0 <= segment_index < len(SEGMENT_REG_ORDER):
            self.runtime.state.segments[SEGMENT_REG_ORDER[segment_index]] = value & 0xFFFF
            return "OK"
        return "E01"

    def monitor_command(self, command: str) -> str:
        parts = command.strip().split()
        if not parts:
            return "E01"
        if parts[0] == "pan-map" and len(parts) in (4, 5):
            try:
                address = int(parts[1], 0)
                size = int(parts[2], 0)
                permissions = parts[3]
                name = parts[4] if len(parts) == 5 else "[pan-monitor]"
                self.runtime.memory.map_region(address, size, permissions, name)
            except (MemoryFault, ValueError):
                return "E14"
            return "OK"
        if parts[0] == "pan-write" and len(parts) == 3:
            try:
                address = int(parts[1], 0)
                data = bytes.fromhex(parts[2])
                self.runtime.memory.write(address, data, record=False)
            except (MemoryFault, ValueError):
                return "E14"
            return "OK"
        return "E01"

    def process(self, payload: str) -> str:
        if payload == "?":
            return self.stop_reply()
        if payload == "g":
            return self.read_registers()
        if payload.startswith("p"):
            return self.read_register(int(payload[1:], 16))
        if payload.startswith("P") and "=" in payload:
            reg_text, data = payload[1:].split("=", 1)
            return self.write_register(int(reg_text, 16), data)
        if payload.startswith("m"):
            address_text, size_text = payload[1:].split(",", 1)
            try:
                data = self.runtime.memory.read(int(address_text, 16), int(size_text, 16), record=False)
            except MemoryFault:
                return "E14"
            return data.hex()
        if payload.startswith("M") and ":" in payload:
            header, data_hex = payload[1:].split(":", 1)
            address_text, size_text = header.split(",", 1)
            data = bytes.fromhex(data_hex)
            if len(data) != int(size_text, 16):
                return "E02"
            try:
                self.runtime.memory.write(int(address_text, 16), data, record=False)
            except MemoryFault:
                return "E14"
            return "OK"
        if payload.startswith("Z0,"):
            address = int(payload.split(",", 2)[1], 16)
            self.runtime.breakpoints.add(address)
            return "OK"
        if payload.startswith("z0,"):
            address = int(payload.split(",", 2)[1], 16)
            self.runtime.breakpoints.discard(address)
            return "OK"
        if payload in {"s", "vCont;s"}:
            self.runtime.step(ignore_breakpoint=True)
            return self.stop_reply()
        if payload in {"c", "vCont;c"}:
            reason = self.runtime.run()
            if reason.signal in (SIGNAL_ILL, SIGNAL_SEGV):
                return self.stop_reply()
            return self.stop_reply()
        if payload.startswith("qSupported"):
            self.no_ack = "QStartNoAckMode+" in payload
            return "PacketSize=4000;qXfer:features:read+;swbreak+;QStartNoAckMode+"
        if payload == "QStartNoAckMode":
            self.no_ack = True
            return "OK"
        if payload.startswith("qAttached"):
            return "1"
        if payload.startswith("qC"):
            return "QC1"
        if payload.startswith("qfThreadInfo"):
            return "m1"
        if payload.startswith("qsThreadInfo"):
            return "l"
        if payload.startswith("qXfer:features:read:target.xml:"):
            _, offset_len = payload.rsplit(":", 1)
            off_text, len_text = offset_len.split(",", 1)
            offset = int(off_text, 16)
            length = int(len_text, 16)
            chunk = REGISTER_XML[offset:offset + length]
            marker = "l" if offset + length >= len(REGISTER_XML) else "m"
            return marker + chunk
        if payload.startswith("qRcmd,"):
            try:
                command = bytes.fromhex(payload.split(",", 1)[1]).decode("utf-8")
            except ValueError:
                return "E01"
            return self.monitor_command(command)
        if payload in {"D", "k"}:
            return "OK"
        return ""


class GdbRemoteServer:
    def __init__(self, runtime: PanRuntime, host: str = "127.0.0.1", port: int = 1234):
        self.runtime = runtime
        self.host = host
        self.port = int(port)
        self.protocol = GdbRemoteProtocol(runtime)

    @classmethod
    def from_endpoint(cls, runtime: PanRuntime, endpoint: str) -> "GdbRemoteServer":
        if endpoint.startswith(":"):
            return cls(runtime, port=int(endpoint[1:]))
        if ":" in endpoint:
            host, port = endpoint.rsplit(":", 1)
            return cls(runtime, host=host, port=int(port))
        return cls(runtime, port=int(endpoint))

    def serve_once(self, *, timeout: Optional[float] = None) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            if timeout is not None:
                server.settimeout(timeout)
            conn, _addr = server.accept()
            with conn:
                conn.settimeout(timeout)
                buf = bytearray()
                while True:
                    byte = conn.recv(1)
                    if not byte:
                        return
                    if byte in (b"+", b"-"):
                        continue
                    if byte != b"$":
                        continue
                    buf.clear()
                    while True:
                        ch = conn.recv(1)
                        if ch == b"#":
                            conn.recv(2)
                            break
                        buf.extend(ch)
                    payload = buf.decode("ascii")
                    if not self.protocol.no_ack:
                        conn.sendall(b"+")
                    response = self.protocol.process(payload)
                    conn.sendall(frame(response))
                    if payload in {"D", "k"}:
                        return
