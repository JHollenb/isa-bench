#include "crt.h"
// LEB128 varint + zigzag encode/decode of 3000 signed 64-bit values with mixed magnitudes; verifies the round trip and
// prints the encoded length, a byte checksum and the number of mismatches.
static unsigned char out[3000 * 10];
static long vals[3000];
static u64 zz(long v) { return ((u64)v << 1) ^ (u64)(v >> 63); }
static long unzz(u64 u) { return (long)(u >> 1) ^ -(long)(u & 1); }
int cmain(void) {
    u64 s = 7;
    for (int i = 0; i < 3000; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        int shift = (int)(s % 60);
        long v = (long)(s >> shift);
        vals[i] = (s & 0x100) ? -v : v;
    }
    u64 n = 0;
    for (int i = 0; i < 3000; ++i) {
        u64 u = zz(vals[i]);
        do { unsigned char b = u & 0x7f; u >>= 7; out[n++] = b | (u ? 0x80 : 0); } while (u);
    }
    u64 pos = 0, bad = 0, chk = 0;
    for (int i = 0; i < 3000; ++i) {
        u64 u = 0; int sh = 0; unsigned char b;
        do { b = out[pos++]; u |= (u64)(b & 0x7f) << sh; sh += 7; } while (b & 0x80);
        if (unzz(u) != vals[i]) ++bad;
    }
    for (u64 i = 0; i < n; ++i) chk = (chk << 5) - chk + out[i];
    put_u64(n); put_u64(chk); put_u64(bad);
    return 0;
}
