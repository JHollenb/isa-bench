#include "crt.h"
// Binary min-heap priority queue (int64 priorities, short payloads): interleaved push/pop, then drain in order.
#define CAP 512
static long pri[CAP];
static short pay[CAP];
static int size;
static void swap(int a, int b) { long t = pri[a]; pri[a] = pri[b]; pri[b] = t; short u = pay[a]; pay[a] = pay[b]; pay[b] = u; }
static void push(long p, short v) {
    int i = size++;
    pri[i] = p; pay[i] = v;
    while (i > 0 && pri[(i - 1) / 2] > pri[i]) { swap(i, (i - 1) / 2); i = (i - 1) / 2; }
}
static long pop(short *v) {
    long top = pri[0];
    *v = pay[0];
    --size;
    pri[0] = pri[size]; pay[0] = pay[size];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < size && pri[l] < pri[m]) m = l;
        if (r < size && pri[r] < pri[m]) m = r;
        if (m == i) break;
        swap(i, m);
        i = m;
    }
    return top;
}
int cmain(void) {
    u64 x = 0x9e3779b97f4a7c15ull, acc = 0, sorted = 1;
    short v;
    for (int i = 0; i < 400; ++i) {
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        push((long)(x * 2685821657736338717ull) >> 20, (short)i);
        if (i % 4 == 3) { long p = pop(&v); acc += (u64)p ^ (u64)v; }
    }
    long prev = -0x7fffffffffffffffL - 1;
    while (size) {
        long p = pop(&v);
        if (p < prev) sorted = 0;
        prev = p;
        acc = acc * 3 + (u64)v;
    }
    put_u64(acc); put_u64(sorted);
    return 0;
}
