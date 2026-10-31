"""Committed SPF v0.2 positive log-add/sub ROM.

The payload is generated with 100-decimal-digit mpmath evaluation and
round-to-nearest, ties-to-even.  Keeping the encoded payload in source makes
the arithmetic reproducible without depending on host libm behavior.
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib

LOGADD_G_ZERO = 2440
LOGADD_ROM_VERSION = "v0.2-rne-mpmath-100dps"
LOGADD_ROM_SHA256 = "67fcf4c5755d502be7483e4c743c9751f9f7a8bc1ee8a7e38b9aece673958066"

_LOGADD_ROM_B85 = (
    b"c-rm31$32V7zglyph&lrAPqWDLb{|80qGvnPNZufjYuO6GLT7kBi-HI&8X2ZVC>%S{XahEIq&(-"
    b"cWd0i6glko^NHaTgM7#bTycZjxs8YOmpqKW;LrIp9?GBcC;Tyg#2@kp@;<-"
    b"E@A43Shu`M6<V}8q2lMOv8ow&9@XP!XzbG&8^ZXn?%g@Nu{1iVaPw?aXm^{jl@Wb*DKgbWr{d^zaEBEl-au?spcgXF0o7~E`$jy9"
    b"{+{ib`^?aRN%h$-i`D(dJuH-A^a=uJ1l}q?yxkxVL3*>w`PtN6Y<ZL-"
    b"h&XhCwUouEemw(D>a;gmEQ{*3VvYaF*$_aA394E)hF><v2yBsA)$`Nw79HtMIL*!ujn?6YXDhKKVWPjOD?<@Q0y=5=`7ui$qAp`X"
    b"8dN<is@1l2>o%D`+2fe-CPPWy5*4yYm>HhkUdTYIvxuxDhZ>~2pH`Ra8o0$Fd#^y$PLvsVYzPX;cuDOn0+g!_B(_F(`-"
    b"Tb||nz^cF74vuI%9h`nD_K^wtY9v0S<bSoWf{xTmfu*Ga#+%`gk^Du#T*v3EaI@R!$J-"
    b"VIxXPvYlr!r=5w0YX&$FvIsMXUZkM@S`nvqW<>xMQy3FBbb~m%RnbqwqZfABoliL|R{LI4)9;WyBQ;*X{Fl_|WL^ySXQ@!FRublF"
    b"gQ+P4CSCe@;sn?T4V&X_n6zK_HYl7Dv|Bc3rf^nnp$5B4czxt#9?1yi0?EmFh@9YoW+cBfRr7@yMkM3n3pMbD{mkGkVhwlrIXDe&"
    b"#Y74UMwuRW@*t6P8+3VXo+6UVM?Mv<3>?iEO_E39FBZ-"
    b"m0@HGk><&7GKpW$zGG5Q!ojIqX4W45uxSYvE8_8Lcxv&L29j`7F{GhoD4@l{flT4hk#l&{LG3aR3%jH;xns#>bP@>9)~ziOvCsqU"
    b"(m>aPZ=VQQ2brzWXD6{KdVd1{ebrdFx7YNOhscBnmSzdEdrt5fQnx~Q(GV0BA{s0Zq?3RN#uxH1%}n23!yh=+tog5*ev)JTW)$cQ"
    b"Y+hMe$4Zsb8e6hI*qL2;BsX_Q5IR6=D`K{eDsE!06hG(aQxp(&c9C0fHDZO|6&(Gi`|72Obkp6G=>=!XFqh(Q>Pp%{)47=_UogK?"
    b"OEiI|Kj2*fl@M-"
    b"XOUCT3#}=3xOAVlkFr8J1%uR$(>PU@g{T12$nZwqP5!V+VF&H}+s3_TvB!;V_Qi7>?rvPU1Aq;4IGJ0xsebF5?QW;W~nG12=ICw-"
    b"JK7xQF|AfQNX5$9RHJJi~LmKp0*k95&ctKmj25Pz;JmA5bicO&`)n6o)>hxD=1#Qv#ll5>aAGLP;qZC8rdWl0M<7C^e;_w3LoM<>"
    b"@H{eMT866VFUpC@W>7>^ukMq|f;m<V(3IH~*5pqC7k=<)i%kYbrnmc_Avyi%?N2#*0%4UXn`jZ>TgcLuGk6D$grWMP7-"
    b"%<(2t&RE1ZiYW#a%ooeu!ycX5wb$DH>$LsS3ydgE>jkzCh!hfKqycuuKTkw{=6>rUd<o^68-iH6o+wyk2J@3Ff@=m-"
    b"n?;^YMZoE4WkUe-${tNFVd-FcLuk6SB%K?0#{FM)qzwyCx2p=kk$>Ds294SZ1-"
    b"}z`cMvj%^<ajwjPLz}6Wci1jA_L`AeVY7JPM1OYUvh>%Q_j+7%Q^a7eV&}JFVGk2i}c0%5`C$@Okb|A&{vvQ>8tg>&1>|v=5^-"
    b"v<_-Er^Ct6V%Pr=u=56NfmOIQlEq7V&w%lX6*K(ibe#-+64>~+#dD!6*hesVAb9&t238yEWo^pEH=^2-2U7mA!-"
    b"sJ_C7u~$%=4H39xOvs>YaU+rFxbNz9^dr%mdCdvd?&&o5x)D1_q=%Dix0f`(94e^@o^+RiR7oRG4wS*d!x_a_=_kS7Uf_5Ys3Guw"
    b"zt;)KWx0C)q5Jz-^yswqIu&VjCU2W"
)
_LOGADD_ROM_BYTES = zlib.decompress(base64.b85decode(_LOGADD_ROM_B85))
_LOGADD_ROM_SHA256_ACTUAL = hashlib.sha256(_LOGADD_ROM_BYTES).hexdigest()
if _LOGADD_ROM_SHA256_ACTUAL != LOGADD_ROM_SHA256:
    raise RuntimeError("SPF v0.2 log-add ROM checksum mismatch")

_expected_bytes = 2 * LOGADD_G_ZERO * struct.calcsize("<h")
if len(_LOGADD_ROM_BYTES) != _expected_bytes:
    raise RuntimeError("SPF v0.2 log-add ROM payload has the wrong size")

_values = struct.unpack(f"<{2 * LOGADD_G_ZERO}h", _LOGADD_ROM_BYTES)
LOGADD_PLUS = tuple(int(value) for value in _values[:LOGADD_G_ZERO])
LOGADD_MINUS = tuple(int(value) for value in _values[LOGADD_G_ZERO:])

if LOGADD_PLUS[0] != 256 or LOGADD_MINUS[1] != -2184:
    raise RuntimeError("SPF v0.2 log-add ROM anchor mismatch")

__all__ = [
    "LOGADD_G_ZERO",
    "LOGADD_MINUS",
    "LOGADD_PLUS",
    "LOGADD_ROM_SHA256",
    "LOGADD_ROM_VERSION",
]
