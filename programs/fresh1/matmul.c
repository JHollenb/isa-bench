#include "crt.h"
// Integer matrix multiply: C = A x B (12x12, int32 entries, int64 accumulate), then trace and a column checksum.
#define D 12
static int A[D][D], B[D][D];
static long C[D][D];
int cmain(void) {
    int s = 7;
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j) {
            s = s * 214013 + 2531011;
            A[i][j] = (s >> 16) % 201 - 100;
            s = s * 214013 + 2531011;
            B[i][j] = (s >> 16) % 61 - 30;
        }
    for (int i = 0; i < D; ++i)
        for (int j = 0; j < D; ++j) {
            long acc = 0;
            for (int k = 0; k < D; ++k) acc += (long)A[i][k] * B[k][j];
            C[i][j] = acc;
        }
    long tr = 0;
    u64 col = 0;
    for (int i = 0; i < D; ++i) tr += C[i][i];
    for (int j = 0; j < D; ++j) {
        long c = 0;
        for (int i = 0; i < D; ++i) c ^= C[i][j] << (i & 7);
        col = col * 31 + (u64)c;
    }
    put_u64((u64)tr); put_u64(col);
    return 0;
}
