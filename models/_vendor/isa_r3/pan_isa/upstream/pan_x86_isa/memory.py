"""Sparse virtual memory for the PAN x86 execution runtime."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


PAGE_SIZE = 4096


class MemoryFault(Exception):
    """Raised when a virtual memory access violates mapping or permissions."""

    def __init__(self, address: int, access: str, reason: str):
        super().__init__(f"{access} fault at 0x{address:x}: {reason}")
        self.address = int(address)
        self.access = access
        self.reason = reason


@dataclass(frozen=True)
class MemoryRegion:
    start: int
    end: int
    permissions: str
    name: str

    def contains(self, address: int, size: int = 1) -> bool:
        return self.start <= address and address + size <= self.end


def align_down(value: int, alignment: int = PAGE_SIZE) -> int:
    return int(value) & ~(alignment - 1)


def align_up(value: int, alignment: int = PAGE_SIZE) -> int:
    return (int(value) + alignment - 1) & ~(alignment - 1)


class SparseMemory:
    """Page-backed virtual memory with region permissions and event logging."""

    def __init__(self, *, page_size: int = PAGE_SIZE):
        self.page_size = int(page_size)
        self._pages: Dict[int, bytearray] = {}
        self.regions: List[MemoryRegion] = []
        self.events: List[Dict[str, object]] = []
        self.dirty_pages: set[int] = set()

    def clone(self) -> "SparseMemory":
        other = SparseMemory(page_size=self.page_size)
        other._pages = {base: bytearray(data) for base, data in self._pages.items()}
        other.regions = list(self.regions)
        other.events = [dict(event) for event in self.events]
        other.dirty_pages = set(self.dirty_pages)
        return other

    def map_region(self, start: int, size: int, permissions: str, name: str = "anon") -> MemoryRegion:
        if size <= 0:
            raise ValueError("memory mapping size must be positive")
        start = align_down(start, self.page_size)
        end = align_up(start + size, self.page_size)
        region = MemoryRegion(start=start, end=end, permissions=permissions, name=name)
        self.regions.append(region)
        self.regions.sort(key=lambda item: item.start)
        for base in range(start, end, self.page_size):
            self._pages.setdefault(base, bytearray(self.page_size))
        return region

    def unmap_region(self, start: int, size: int) -> None:
        start = align_down(start, self.page_size)
        end = align_up(start + size, self.page_size)
        self.regions = [region for region in self.regions if region.end <= start or region.start >= end]
        for base in range(start, end, self.page_size):
            self._pages.pop(base, None)
            self.dirty_pages.discard(base)

    def protect_region(self, start: int, size: int, permissions: str) -> None:
        start = align_down(start, self.page_size)
        end = align_up(start + size, self.page_size)
        changed = False
        next_regions: List[MemoryRegion] = []
        for region in self.regions:
            if region.start == start and region.end == end:
                next_regions.append(MemoryRegion(start, end, permissions, region.name))
                changed = True
            else:
                next_regions.append(region)
        if not changed:
            raise MemoryFault(start, "mprotect", "mapping not found")
        self.regions = sorted(next_regions, key=lambda item: item.start)

    def find_region(self, address: int, size: int = 1) -> Optional[MemoryRegion]:
        for region in self.regions:
            if region.contains(address, size):
                return region
        return None

    def check(self, address: int, size: int, access: str) -> MemoryRegion:
        if address < 0 or address >= (1 << 64) or address + size > (1 << 64):
            raise MemoryFault(address, access, "noncanonical or out-of-range address")
        region = self.find_region(address, size)
        if region is None:
            raise MemoryFault(address, access, "unmapped")
        permission = {"read": "r", "write": "w", "execute": "x"}[access]
        if permission not in region.permissions:
            raise MemoryFault(address, access, f"missing {permission} permission")
        return region

    def read(self, address: int, size: int, *, access: str = "read", record: bool = True) -> bytes:
        self.check(address, size, access)
        out = bytearray()
        remaining = int(size)
        cursor = int(address)
        while remaining:
            page_base = align_down(cursor, self.page_size)
            page_offset = cursor - page_base
            chunk = min(remaining, self.page_size - page_offset)
            page = self._pages.get(page_base)
            if page is None:
                raise MemoryFault(cursor, access, "unmapped page")
            out.extend(page[page_offset:page_offset + chunk])
            cursor += chunk
            remaining -= chunk
        data = bytes(out)
        if record:
            self.events.append({"kind": access, "address": hex(address), "size": size, "value_hex": data.hex()})
        return data

    def write(self, address: int, data: bytes, *, record: bool = True) -> None:
        self.check(address, len(data), "write")
        remaining = len(data)
        cursor = int(address)
        src = 0
        while remaining:
            page_base = align_down(cursor, self.page_size)
            page_offset = cursor - page_base
            chunk = min(remaining, self.page_size - page_offset)
            page = self._pages.get(page_base)
            if page is None:
                raise MemoryFault(cursor, "write", "unmapped page")
            page[page_offset:page_offset + chunk] = data[src:src + chunk]
            self.dirty_pages.add(page_base)
            cursor += chunk
            src += chunk
            remaining -= chunk
        if record:
            self.events.append({"kind": "write", "address": hex(address), "size": len(data), "value_hex": data.hex()})

    def load_bytes(self, address: int, data: bytes) -> None:
        self.write(address, data, record=False)

    def read_int(self, address: int, size: int, *, signed: bool = False) -> int:
        return int.from_bytes(self.read(address, size), "little", signed=signed)

    def write_int(self, address: int, value: int, size: int) -> None:
        self.write(address, int(value).to_bytes(size, "little", signed=False))

    def read_c_string(self, address: int, limit: int = 1 << 20) -> bytes:
        out = bytearray()
        for offset in range(limit):
            byte = self.read(address + offset, 1, record=False)
            if byte == b"\x00":
                return bytes(out)
            out.extend(byte)
        raise MemoryFault(address, "read", "unterminated string")

    def mapped_ranges(self) -> List[Dict[str, object]]:
        return [
            {
                "start": hex(region.start),
                "end": hex(region.end),
                "permissions": region.permissions,
                "name": region.name,
            }
            for region in self.regions
        ]

    def digest(self, pages: Optional[Iterable[int]] = None) -> str:
        h = hashlib.sha256()
        selected = sorted(self._pages if pages is None else pages)
        for page_base in selected:
            page = self._pages.get(page_base)
            if page is None:
                continue
            h.update(int(page_base).to_bytes(8, "little"))
            h.update(bytes(page))
        return "sha256:" + h.hexdigest()
