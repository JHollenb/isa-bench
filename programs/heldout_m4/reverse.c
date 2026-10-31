#include "crt.h"
// Reverse words and whole strings in place; palindrome check; print one reversed sentence and two counters.
static char buf[2048];
static long slen(const char *s) { long n = 0; while (s[n]) ++n; return n; }
static void rev(char *a, char *b) { while (a < b) { char t = *a; *a++ = *b; *b-- = t; } }
static int is_pal(const char *s, long n) { for (long i = 0, j = n - 1; i < j; ++i, --j) if (s[i] != s[j]) return 0; return 1; }
int cmain(void) {
    const char *src = "the quick brown fox jumps over the lazy dog";
    long n = slen(src);
    for (long i = 0; i <= n; ++i) buf[i] = src[i];
    rev(buf, buf + n - 1);
    for (long i = 0, s = 0; i <= n; ++i) if (buf[i] == ' ' || buf[i] == 0) { rev(buf + s, buf + i - 1); s = i + 1; }
    put(buf, n); put("\n", 1);
    u64 pals = 0, total = 0;
    for (int k = 0; k < 300; ++k) {
        long m = 3 + k % 29;
        for (long i = 0; i < m; ++i) buf[i] = (char)('a' + ((k * 7 + i * (i % 3 == 0 ? 0 : 1)) % 3));
        buf[m] = 0;
        rev(buf, buf + m - 1);
        pals += is_pal(buf, m);
        total += (unsigned char)buf[0];
    }
    put_u64(pals); put_u64(total);
    return 0;
}
