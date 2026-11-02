#include <stdio.h>
#include <stdlib.h>
int main(void) {
    const char *in[] = {"42", "-17", "  300", "0x1f", "999999", "12abc"};
    long t = 0;
    for (int i = 0; i < 6; i++) t += atoi(in[i]) + strtol(in[i], 0, 0);
    printf("%ld\n", t);
    return 0;
}
