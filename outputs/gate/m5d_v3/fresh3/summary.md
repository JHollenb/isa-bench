# Proof gate: m5d_v3 on fresh3

Prove job job-eecf1edab58a, run job job-552169eb394f, certificate sha256 738f83d817af.

Static instructions: proved 2993, refuted 1, no learned record 105, OS boundary 37.

| arm | complete (all steps exact) | stdout ok | exact steps / native | learned steps | hybrid steps |
|---|---:|---:|---:|---:|---:|
| G0-m4+program | 11/12 | 11/12 | 14937015/15200920 | 13894186 | 1042782 |
| H0-proved | 11/12 | 11/12 | 14937015/15200920 | 0 | 14936968 |
| M4-proved-only | 0/12 | 0/12 | 976889/15200920 | 976877 | 0 |
| M4-unguarded | 0/12 | 0/12 | 976889/15200920 | 979128 | 0 |

| program | G0-m4+program | H0-proved | M4-proved-only | M4-unguarded |
|---|---|---|---|---|
| crc32-O0 | exit | exit | stop: no proved path at 0x401314 (add) | stop: no proved path at 0x401314 (add) |
| crc32-O2 | stop: no proved path at 0x4010da (paddd) | stop: no proved path at 0x4010da (paddd) | stop: no proved path at 0x4010a1 (movdqa) | stop: no proved path at 0x4010a1 (movdqa) |
| dijkstra-O0 | exit | exit | stop: no proved path at 0x401342 (mov) | stop: no proved path at 0x401342 (mov) |
| dijkstra-O2 | exit | exit | stop: no proved path at 0x40112f (lea) | stop: no proved path at 0x40112f (lea) |
| lcs-O0 | exit | exit | stop: no proved path at 0x4012b3 (mov) | stop: no proved path at 0x4012b3 (mov) |
| lcs-O2 | exit | exit | stop: no proved path at 0x401118 (test) | stop: no proved path at 0x40113f (mov) |
| modpow-O0 | exit | exit | stop: no proved path at 0x4012e1 (test) | stop: no proved path at 0x4012e1 (test) |
| modpow-O2 | exit | exit | stop: no proved path at 0x4010e8 (mov) | stop: no proved path at 0x4010e8 (mov) |
| partition-O0 | exit | exit | stop: no proved path at 0x401262 (mov) | stop: no proved path at 0x401262 (mov) |
| partition-O2 | exit | exit | stop: no proved path at 0x4010be (mov) | stop: no proved path at 0x4010be (mov) |
| varint-O0 | exit | exit | stop: no proved path at 0x40127a (test) | stop: no proved path at 0x40127a (test) |
| varint-O2 | exit | exit | stop: no proved path at 0x40114d (lea) | stop: no proved path at 0x40114d (lea) |
