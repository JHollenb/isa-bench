#include "crt.h"
// Three-way (Dutch flag) quicksort of 4000 records keyed by a signed 32-bit key with many duplicates, iterative with
// an explicit stack; verifies order and prints a key checksum, the median key and distinct-key count.
struct rec { int key; unsigned short tag; unsigned char flag; };
static struct rec r[4000];
static int stk[256];
static void swp(int i, int j) { struct rec t = r[i]; r[i] = r[j]; r[j] = t; }
int cmain(void) {
    unsigned s = 77;
    for (int i = 0; i < 4000; ++i) {
        s = s * 22695477u + 1u;
        r[i].key = (int)((s >> 8) % 401) - 200;
        r[i].tag = (unsigned short)i;
        r[i].flag = (unsigned char)(s & 3);
    }
    int sp = 0;
    stk[sp++] = 0; stk[sp++] = 3999;
    while (sp) {
        int hi = stk[--sp], lo = stk[--sp];
        if (lo >= hi) continue;
        int pv = r[lo + (hi - lo) / 2].key, lt = lo, i = lo, gt = hi;
        while (i <= gt) {
            if (r[i].key < pv) swp(lt++, i++);
            else if (r[i].key > pv) swp(i, gt--);
            else ++i;
        }
        stk[sp++] = lo; stk[sp++] = lt - 1;
        stk[sp++] = gt + 1; stk[sp++] = hi;
    }
    u64 chk = 0, bad = 0, distinct = 1;
    for (int i = 0; i < 4000; ++i) {
        chk = chk * 33 + (u64)(long)r[i].key + r[i].flag;
        if (i && r[i - 1].key > r[i].key) ++bad;
        if (i && r[i - 1].key != r[i].key) ++distinct;
    }
    put_u64(chk); put_u64((u64)(long)r[2000].key); put_u64(distinct); put_u64(bad);
    return 0;
}
