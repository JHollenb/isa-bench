#include <stdio.h>
#include <stdlib.h>
static int cmp(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }
int main(void) {
    int v[64];
    unsigned s = 12345;
    for (int i = 0; i < 64; i++) { s = s * 1103515245u + 12345u; v[i] = (int)(s >> 16) % 1000; }
    qsort(v, 64, sizeof v[0], cmp);
    long chk = 0;
    for (int i = 0; i < 64; i++) chk = chk * 31 + v[i];
    printf("%d %d %ld\n", v[0], v[63], chk);
    return 0;
}
