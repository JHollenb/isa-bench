#include "crt.h"
// Multi-word (32-bit limbs) addition and doubling: Fibonacci numbers to F(800) in 640-bit limbs, digit sum.
#define N 20
static unsigned int A[N], B[N], C[N];
static void add(unsigned int *r, const unsigned int *x, const unsigned int *y) {
    u64 carry = 0;
    for (int i = 0; i < N; ++i) { u64 s = (u64)x[i] + y[i] + carry; r[i] = (unsigned int)s; carry = s >> 32; }
}
static unsigned int divmod10(unsigned int *x) {
    u64 rem = 0;
    for (int i = N - 1; i >= 0; --i) { u64 cur = (rem << 32) | x[i]; x[i] = (unsigned int)(cur / 10); rem = cur % 10; }
    return (unsigned int)rem;
}
static int is_zero(const unsigned int *x) { for (int i = 0; i < N; ++i) if (x[i]) return 0; return 1; }
int cmain(void) {
    A[0] = 0; B[0] = 1;
    for (int k = 0; k < 800; ++k) { add(C, A, B); for (int i = 0; i < N; ++i) { A[i] = B[i]; B[i] = C[i]; } }
    u64 top = 0;
    for (int i = N - 1; i >= 0; --i) if (A[i]) { top = (u64)i; break; }
    u64 digits = 0, dsum = 0;
    while (!is_zero(A)) { dsum += divmod10(A); ++digits; }
    put_u64(top); put_u64(digits); put_u64(dsum);
    return 0;
}
