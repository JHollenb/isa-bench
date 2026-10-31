#include "crt.h"
// Glob matcher ('*', '?', '[a-z]' classes) by backtracking, over a fixed pattern x subject grid.
static int cls(const char **pp, char c) {
    const char *p = *pp + 1;
    int neg = 0, ok = 0;
    if (*p == '!') { neg = 1; ++p; }
    while (*p && *p != ']') {
        if (p[1] == '-' && p[2] && p[2] != ']') { if (c >= p[0] && c <= p[2]) ok = 1; p += 3; }
        else { if (c == *p) ok = 1; ++p; }
    }
    *pp = *p ? p + 1 : p;
    return ok ^ neg;
}
static int match(const char *p, const char *s) {
    const char *star = 0, *ss = 0;
    while (*s) {
        if (*p == '*') { star = ++p; ss = s; continue; }
        if (*p == '[') { const char *q = p; if (cls(&q, *s)) { p = q; ++s; continue; } }
        else if (*p == '?' || *p == *s) { ++p; ++s; continue; }
        if (!star) return 0;
        p = star; s = ++ss;
    }
    while (*p == '*') ++p;
    return *p == 0;
}
static const char *pats[] = {"*.c", "a*b?c", "[a-m]*z", "*[!0-9]", "??*??", "*x*y*z*", "[0-9][0-9]*", "a[bc]*[de]f"};
static const char *subs[] = {"main.c", "aXXbYc", "hazz", "file9", "ab", "xxyyzz", "42abc", "abdef", "acccef", "zebra.c",
                             "mz", "a1b2c", "xyz", "12", "abef", "test.h"};
int cmain(void) {
    u64 bits = 0, n = 0;
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 16; ++j) {
            int m = match(pats[i], subs[j]);
            bits = (bits << 1 | (u64)m) ^ (bits >> 61);
            n += (u64)m;
        }
    put_u64(n); put_u64(bits);
    return 0;
}
