# Native C/assembly timing baseline

`scripts/ssm_campaign.py` produces identical Linux x86-64 ELF bytes for all six learned decoders and direct native execution on Beast. Each requested length counts the entire executed instruction path, including initialization and exit. The counter explicitly initializes RDI, adds a known count and exits with its low byte. The stdout program computes a value, writes eight bytes and exits. ELF files, `.S` byte sources and SHA-256 identities are sealed into each campaign.

Native subprocess measurements include process creation, ELF loading, stdout/stderr capture and wait. They do not produce the emulator's architectural state and mapped-page hashes. Compare these to model time-to-stop with that API boundary in view; full diagnostic response timing is reported separately. Native correctness uses observable stdout, stderr and exit status. Nonzero expected counter exits are successful results.

`kernel.c` adds a second, separate resident arithmetic floor. Compile with `cc -O3 -std=c11`; volatile inline assembly fixes dependent ADD chains and prevents constant folding. Each size performs 20,000 calls per sample with a checked checksum. Kernel timing excludes ELF startup, system calls and diagnostics, so it is not an interchangeable end-to-end result.

The baseline has no model training. Compiler identity, flags, build wall/CPU time, source/binary hashes and binary size are recorded as build costs instead.
