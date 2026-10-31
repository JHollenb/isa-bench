# Proof gate: m5d_v3 on c_apps

Prove job job-f43c311f9544, run job job-8202bffa8670, certificate sha256 b73b59631345.

Static instructions: proved 215, refuted 0, no learned record 17, OS boundary 8.

| arm | complete (all steps exact) | stdout ok | exact steps / native | learned steps | hybrid steps |
|---|---:|---:|---:|---:|---:|
| G0-m4+program | 3/4 | 3/4 | 3317/15544 | 3233 | 77 |
| H0-proved | 2/4 | 2/4 | 201/15544 | 0 | 195 |
| M4-proved-only | 1/4 | 1/4 | 48/15544 | 43 | 0 |
| M4-unguarded | 1/4 | 1/4 | 48/15544 | 43 | 0 |

| program | G0-m4+program | H0-proved | M4-proved-only | M4-unguarded |
|---|---|---|---|---|
| large_records-Os | stop: no proved path at 0x40106b (inc) | stop: no proved path at 0x401019 (inc) | stop: no proved path at 0x40104f (lea) | stop: no proved path at 0x40104f (lea) |
| medium_sort-Os | exit | stop: no proved path at 0x401015 (inc) | stop: no proved path at 0x401011 (mov) | stop: no proved path at 0x401011 (mov) |
| small_checksum-Os | exit | exit | stop: no proved path at 0x401067 (lea) | stop: no proved path at 0x401067 (lea) |
| tiny_hello-Os | exit | exit | exit | exit |
