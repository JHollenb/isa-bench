#include "crt.h"
// Open-addressing hash table (linear probing): insert 600 keys, look up 1200 (half present), delete 200, re-probe.
#define CAP 1024
static u64 keys[CAP];
static long vals[CAP];
static unsigned char used[CAP];
static u64 mix(u64 k) { k ^= k >> 29; k *= 0xbf58476d1ce4e5b9ull; k ^= k >> 32; return k; }
static void insert(u64 k, long v) {
    u64 i = mix(k) & (CAP - 1);
    while (used[i] == 1 && keys[i] != k) i = (i + 1) & (CAP - 1);
    keys[i] = k; vals[i] = v; used[i] = 1;
}
static long find(u64 k) {
    u64 i = mix(k) & (CAP - 1);
    for (int n = 0; n < CAP && used[i]; ++n) {
        if (used[i] == 1 && keys[i] == k) return vals[i];
        i = (i + 1) & (CAP - 1);
    }
    return -1;
}
static void erase(u64 k) {
    u64 i = mix(k) & (CAP - 1);
    for (int n = 0; n < CAP && used[i]; ++n) {
        if (used[i] == 1 && keys[i] == k) { used[i] = 2; return; }
        i = (i + 1) & (CAP - 1);
    }
}
int cmain(void) {
    u64 x = 99;
    for (int i = 0; i < 600; ++i) { x = x * 2862933555777941757ull + 3037000493ull; insert(x >> 7, i * 3 + 1); }
    u64 hits = 0, sum = 0;
    x = 99;
    for (int i = 0; i < 1200; ++i) {
        x = x * 2862933555777941757ull + 3037000493ull;
        long v = find((i & 1) ? (x >> 7) + 1 : (x >> 7));
        if (v >= 0) { ++hits; sum += (u64)v; }
    }
    x = 99;
    for (int i = 0; i < 200; ++i) { x = x * 2862933555777941757ull + 3037000493ull; erase(x >> 7); }
    u64 after = 0;
    x = 99;
    for (int i = 0; i < 600; ++i) { x = x * 2862933555777941757ull + 3037000493ull; after += find(x >> 7) >= 0; }
    put_u64(hits); put_u64(sum); put_u64(after);
    return 0;
}
