#include "crt.h"
// Chained hash map with string keys (FNV-1a, 128 buckets, nodes from a static pool): counts word frequencies in a
// generated text, then looks up a fixed list and reports the most frequent word's length and count.
#define NB 128
#define POOL 1024
struct node { struct node *next; const char *word; int len; int count; };
static struct node pool[POOL], *buckets[NB];
static int used;
static char text[6000];
static unsigned fnv(const char *s, int n) { unsigned h = 2166136261u; for (int i = 0; i < n; ++i) h = (h ^ (unsigned char)s[i]) * 16777619u; return h; }
static int eq(const char *a, const char *b, int n) { for (int i = 0; i < n; ++i) if (a[i] != b[i]) return 0; return 1; }
static struct node *get(const char *w, int n, int create) {
    unsigned b = fnv(w, n) & (NB - 1);
    for (struct node *p = buckets[b]; p; p = p->next)
        if (p->len == n && eq(p->word, w, n)) return p;
    if (!create || used == POOL) return 0;
    struct node *p = &pool[used++];
    p->word = w; p->len = n; p->count = 0; p->next = buckets[b]; buckets[b] = p;
    return p;
}
int cmain(void) {
    static const char *syl[] = {"ka", "lo", "mi", "ra", "ten", "su", "vo", "ne"};
    unsigned s = 4242;
    int t = 0;
    while (t < 5900) {
        s = s * 1103515245u + 12345u;
        int ns = 1 + (int)((s >> 16) % 3);
        for (int k = 0; k < ns; ++k) {
            s = s * 1103515245u + 12345u;
            for (const char *c = syl[(s >> 20) & 7]; *c; ++c) text[t++] = *c;
        }
        text[t++] = ' ';
    }
    int i = 0, words = 0;
    while (i < t) {
        while (i < t && text[i] == ' ') ++i;
        int j = i;
        while (j < t && text[j] != ' ') ++j;
        if (j > i) { struct node *p = get(text + i, j - i, 1); if (p) ++p->count; ++words; }
        i = j;
    }
    struct node *best = 0;
    for (int k = 0; k < used; ++k) if (!best || pool[k].count > best->count) best = &pool[k];
    u64 found = 0;
    static const char *q[] = {"ka", "lomi", "tensu", "vone", "rakamo", "su"};
    for (int k = 0; k < 6; ++k) { int n = 0; while (q[k][n]) ++n; struct node *p = get(q[k], n, 0); found = found * 1000 + (u64)(p ? p->count : 0); }
    put_u64((u64)words); put_u64((u64)used); put_u64((u64)best->len * 1000 + (u64)best->count); put_u64(found);
    return 0;
}
