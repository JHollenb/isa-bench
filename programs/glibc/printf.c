#include <stdio.h>
int main(void) {
    long s = 0;
    for (int i = 1; i <= 20; i++) s += (long)i * i;
    printf("sum=%ld hex=%#lx neg=%d str=%s\n", s, (unsigned long)s, -42, "ok");
    return 0;
}
