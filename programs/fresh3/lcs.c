#include "crt.h"
// Longest common subsequence of two generated 250-character strings over a 4-letter alphabet with a two-row DP
// table, plus the edit (Levenshtein) distance of their first 150 characters.
#define N 250
static char a[N], b[N];
static int row0[N + 1], row1[N + 1];
static int min3(int x, int y, int z) { int m = x < y ? x : y; return m < z ? m : z; }
int cmain(void) {
    unsigned s = 2024;
    for (int i = 0; i < N; ++i) { s = s * 1664525u + 1013904223u; a[i] = "ACGT"[s >> 30]; }
    for (int i = 0; i < N; ++i) { s = s * 1664525u + 1013904223u; b[i] = (s & 0x1000) ? a[i] : "ACGT"[s >> 30]; }
    int *p = row0, *q = row1;
    for (int j = 0; j <= N; ++j) p[j] = 0;
    for (int i = 1; i <= N; ++i) {
        q[0] = 0;
        for (int j = 1; j <= N; ++j)
            q[j] = a[i - 1] == b[j - 1] ? p[j - 1] + 1 : (p[j] > q[j - 1] ? p[j] : q[j - 1]);
        int *t = p; p = q; q = t;
    }
    int lcs = p[N];
    const int M = 150;
    for (int j = 0; j <= M; ++j) p[j] = j;
    for (int i = 1; i <= M; ++i) {
        q[0] = i;
        for (int j = 1; j <= M; ++j) q[j] = min3(p[j] + 1, q[j - 1] + 1, p[j - 1] + (a[i - 1] != b[j - 1]));
        int *t = p; p = q; q = t;
    }
    put_u64((u64)lcs); put_u64((u64)p[M]);
    return 0;
}
