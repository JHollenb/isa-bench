#include "crt.h"
static u64 len(const char *s) { u64 n = 0; while (s[n]) ++n; return n; }
static int cmp(const char *a, const char *b) { while (*a && *a == *b) { ++a; ++b; } return (unsigned char)*a - (unsigned char)*b; }
static void reverse(char *s, u64 n) { for (u64 i = 0; i < n / 2; ++i) { char t = s[i]; s[i] = s[n - 1 - i]; s[n - 1 - i] = t; } }
static void upper(char *s) { for (; *s; ++s) if (*s >= 'a' && *s <= 'z') *s -= 32; }
static char buf[2048];
int cmain(void) {
    const char *word = "the quick brown fox jumps over the lazy dog ";
    u64 w = len(word), n = 0;
    for (int r = 0; r < 40; ++r) for (u64 i = 0; i < w; ++i) buf[n++] = word[i];
    buf[n] = 0;
    u64 h = 0;
    for (int r = 0; r < 4; ++r) { reverse(buf, n); upper(buf); h += (u64)cmp(buf, word) + len(buf); }
    put_u64(h);
    return 0;
}
