#include "crt.h"
static long v[512];
static void quick(long *p, long n) {
    while (n > 16) {
        long pivot = p[n / 2], i = 0, j = n - 1;
        while (i <= j) {
            while (p[i] < pivot) ++i;
            while (p[j] > pivot) --j;
            if (i <= j) { long t = p[i]; p[i] = p[j]; p[j] = t; ++i; --j; }
        }
        if (j + 1 < n - i) { quick(p, j + 1); p += i; n -= i; } else { quick(p + i, n - i); n = j + 1; }
    }
    for (long i = 1; i < n; ++i) { long t = p[i], k = i; while (k > 0 && p[k - 1] > t) { p[k] = p[k - 1]; --k; } p[k] = t; }
}
int cmain(void) {
    u64 x = 12345;
    for (int i = 0; i < 512; ++i) { x = x * 6364136223846793005ull + 1442695040888963407ull; v[i] = (long)(x >> 33) - (1l << 30); }
    quick(v, 512);
    u64 ok = 1, h = 0;
    for (int i = 1; i < 512; ++i) ok &= v[i - 1] <= v[i];
    for (int i = 0; i < 512; ++i) h = h * 31 + (u64)v[i];
    put_u64(ok); put_u64(h);
    return 0;
}
