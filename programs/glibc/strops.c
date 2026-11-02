#include <stdio.h>
#include <string.h>
static char buf[256], dst[256];
int main(void) {
    const char *words[] = {"alpha", "bravo", "charlie", "delta", "echo"};
    size_t total = 0;
    for (int i = 0; i < 5; i++) {
        strcpy(buf, words[i]);
        strcat(buf, "-x");
        memcpy(dst, buf, strlen(buf) + 1);
        total += strlen(dst) + (strcmp(dst, "charlie-x") == 0) + (memchr(dst, 'h', strlen(dst)) != 0);
    }
    memset(buf, 'z', 100);
    buf[100] = 0;
    printf("%zu %zu %d\n", total, strlen(buf), memcmp(buf, dst, 3) > 0);
    return 0;
}
