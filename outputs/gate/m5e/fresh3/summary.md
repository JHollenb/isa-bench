# Proof gate: m5e on fresh3

Prove job job-066492a2b656, run job job-6016a61ebc49, certificate sha256 a873e2838417.

Static instructions: proved 3067, refuted 0, no learned record 32, OS boundary 37.

| arm | complete (all steps exact) | stdout ok | exact steps / native | learned steps | hybrid steps |
|---|---:|---:|---:|---:|---:|
| G0-m4+program | 11/12 | 11/12 | 14937015/15200920 | 14641092 | 295876 |
| H0-proved | 11/12 | 11/12 | 14937015/15200920 | 0 | 14936968 |
| M4-proved-only | 4/12 | 4/12 | 5794217/15200920 | 5794192 | 0 |
| M4-unguarded | 4/12 | 4/12 | 5794217/15200920 | 5794192 | 0 |

| program | G0-m4+program | H0-proved | M4-proved-only | M4-unguarded |
|---|---|---|---|---|
| crc32-O0 | exit | exit | exit | exit |
| crc32-O2 | stop: no proved path at 0x4010da (paddd) | stop: no proved path at 0x4010da (paddd) | stop: no proved path at 0x4010a1 (movdqa) | stop: no proved path at 0x4010a1 (movdqa) |
| dijkstra-O0 | exit | exit | exit | exit |
| dijkstra-O2 | exit | exit | exit | exit |
| lcs-O0 | exit | exit | stop: no proved path at 0x401353 (cmovge) | stop: no proved path at 0x401353 (cmovge) |
| lcs-O2 | exit | exit | stop: no proved path at 0x401118 (test) | stop: no proved path at 0x401118 (test) |
| modpow-O0 | exit | exit | exit | exit |
| modpow-O2 | exit | exit | stop: no proved path at 0x401150 (cmovae) | stop: no proved path at 0x401150 (cmovae) |
| partition-O0 | exit | exit | stop: no proved path at 0x401262 (mov) | stop: no proved path at 0x401262 (mov) |
| partition-O2 | exit | exit | stop: no proved path at 0x4010be (mov) | stop: no proved path at 0x4010be (mov) |
| varint-O0 | exit | exit | stop: no proved path at 0x40137c (shl) | stop: no proved path at 0x40137c (shl) |
| varint-O2 | exit | exit | stop: no proved path at 0x4011b2 (test) | stop: no proved path at 0x4011b2 (test) |
