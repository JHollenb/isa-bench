#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    char *p[16];
    size_t sum = 0;
    for (int i = 0; i < 16; i++) { p[i] = malloc(16 + 40 * i); memset(p[i], i, 16 + 40 * i); }
    for (int i = 0; i < 16; i += 2) { free(p[i]); p[i] = 0; }
    for (int i = 0; i < 16; i++) if (p[i]) sum += (unsigned char)p[i][5] + 1;
    char *q = calloc(100, 4);
    q = realloc(q, 1000);
    q[999] = 7;
    sum += q[0] + q[999];
    printf("%zu\n", sum);
    return 0;
}
