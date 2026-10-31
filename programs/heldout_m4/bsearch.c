#include "crt.h"
// Binary search over a sorted array of 4096 32-bit values; 3000 queries, count found and sum of positions.
static unsigned int a[4096];
static long lower_bound(const unsigned int *p, long n, unsigned int key) {
    long lo = 0, hi = n;
    while (lo < hi) {
        long mid = lo + ((hi - lo) >> 1);
        if (p[mid] < key) lo = mid + 1; else hi = mid;
    }
    return lo;
}
int cmain(void) {
    unsigned int v = 7;
    for (int i = 0; i < 4096; ++i) { v += 1 + ((v * 2654435761u) >> 28); a[i] = v; }
    u64 found = 0, pos = 0;
    unsigned int q = 12345;
    for (int i = 0; i < 3000; ++i) {
        q = q * 1103515245u + 12345u;
        unsigned int key = q % (v + 1);
        long p = lower_bound(a, 4096, key);
        if (p < 4096 && a[p] == key) ++found;
        pos += (u64)p;
    }
    put_u64(found); put_u64(pos);
    return 0;
}
