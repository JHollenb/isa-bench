#include "crt.h"
static unsigned char a[4096], b[4096];
int cmain(void) {
    u64 x = 88172645463325252ull, sum = 0;
    for (int i = 0; i < 4096; ++i) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; a[i] = (unsigned char)x; }
    for (int r = 0; r < 8; ++r) {
        memcpy(b + r, a, 4096 - r);
        for (int i = 0; i < 4096; i += 64) sum += b[i];
    }
    put_u64(sum);
    return 0;
}
