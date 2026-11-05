#include "crt.h"
// Number theory on signed and unsigned 64-bit integers: binary GCD and extended Euclid (modular inverses),
// Miller-Rabin primality for 100 odd candidates with 128-bit-free mulmod (double-and-add), and a signed
// floor-division table.
static u64 mulmod(u64 a, u64 b, u64 m) {
    u64 r = 0; a %= m;
    while (b) { if (b & 1) { r += a; if (r >= m) r -= m; } a <<= 1; if (a >= m) a -= m; b >>= 1; }
    return r;
}
static u64 powmod(u64 a, u64 e, u64 m) { u64 r = 1; while (e) { if (e & 1) r = mulmod(r, a, m); a = mulmod(a, a, m); e >>= 1; } return r; }
static int prime(u64 n) {
    if (n < 4) return n > 1;
    u64 d = n - 1; int s = 0;
    while (!(d & 1)) { d >>= 1; ++s; }
    static const u64 bases[] = {2, 3, 5, 7, 11, 13};
    for (int k = 0; k < 6; ++k) {
        u64 a = bases[k] % n; if (!a) continue;
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        int ok = 0;
        for (int i = 1; i < s; ++i) { x = mulmod(x, x, n); if (x == n - 1) { ok = 1; break; } }
        if (!ok) return 0;
    }
    return 1;
}
static long inv(long a, long m) {
    long t = 0, nt = 1, r = m, nr = a % m;
    while (nr) { long q = r / nr, tmp = t - q * nt; t = nt; nt = tmp; tmp = r - q * nr; r = nr; nr = tmp; }
    return r > 1 ? -1 : (t < 0 ? t + m : t);
}
int cmain(void) {
    u64 cnt = 0, last = 0, cand = 1000003ULL;
    for (int i = 0; i < 100; ++i, cand += 2) if (prime(cand)) { ++cnt; last = cand; }
    u64 isum = 0;
    for (long a = 2; a < 300; ++a) { long v = inv(a, 1000003); isum = isum * 7 + (u64)(v < 0 ? 0 : v); }
    long fsum = 0;
    for (long x = -50; x <= 50; ++x)
        for (long y = -7; y <= 7; ++y)
            if (y) { long q = x / y; if ((x % y != 0) && ((x < 0) != (y < 0))) --q; fsum = fsum * 3 + q; }
    put_u64(cnt); put_u64(last); put_u64(isum); put_u64((u64)fsum);
    return 0;
}
