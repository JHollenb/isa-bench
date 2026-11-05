#include "crt.h"
// Table-driven CRC-32 (reflected, poly 0xEDB88320) over a generated 6000-byte buffer, plus per-block CRCs and an
// Adler-32 of the same data.
typedef unsigned int u32;
static u32 table[256];
static unsigned char buf[6000];
int cmain(void) {
    for (u32 i = 0; i < 256; ++i) {
        u32 c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        table[i] = c;
    }
    u32 x = 12345;
    for (int i = 0; i < 6000; ++i) { x = x * 1103515245u + 12345u; buf[i] = (unsigned char)(x >> 16); }
    u32 crc = 0xFFFFFFFFu;
    for (int i = 0; i < 6000; ++i) crc = table[(crc ^ buf[i]) & 0xFF] ^ (crc >> 8);
    crc ^= 0xFFFFFFFFu;
    u64 blocks = 0;
    for (int b = 0; b < 6000; b += 500) {
        u32 c = 0xFFFFFFFFu;
        for (int i = b; i < b + 500; ++i) c = table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
        blocks = blocks * 131 + (c ^ 0xFFFFFFFFu);
    }
    u32 a = 1, s = 0;
    for (int i = 0; i < 6000; ++i) { a = (a + buf[i]) % 65521u; s = (s + a) % 65521u; }
    put_u64(crc); put_u64(blocks); put_u64(((u64)s << 16) | a);
    return 0;
}
