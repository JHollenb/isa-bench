# Proof gate: m5e on c_apps

Prove job job-7e05af81f845, run job job-38d3c787ee61, certificate sha256 70d5b17c5460.

Static instructions: proved 232, refuted 0, no learned record 0, OS boundary 8.

| arm | complete (all steps exact) | stdout ok | exact steps / native | learned steps | hybrid steps |
|---|---:|---:|---:|---:|---:|
| G0-m4+program | 4/4 | 4/4 | 15544/15544 | 15536 | 0 |
| H0-proved | 2/4 | 2/4 | 201/15544 | 0 | 195 |
| M4-proved-only | 4/4 | 4/4 | 15544/15544 | 15536 | 0 |
| M4-unguarded | 4/4 | 4/4 | 15544/15544 | 15536 | 0 |

| program | G0-m4+program | H0-proved | M4-proved-only | M4-unguarded |
|---|---|---|---|---|
| large_records-Os | exit | stop: no proved path at 0x401019 (inc) | exit | exit |
| medium_sort-Os | exit | stop: no proved path at 0x401015 (inc) | exit | exit |
| small_checksum-Os | exit | exit | exit | exit |
| tiny_hello-Os | exit | exit | exit | exit |
