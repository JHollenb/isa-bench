#include "crt.h"
// Adler-32, CRC-32 (bitwise) and Fletcher-16 over a generated 3000-byte buffer.
static unsigned char data[3000];
int cmain(void) {
    unsigned int x = 1;
    for (int i = 0; i < 3000; ++i) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; data[i] = (unsigned char)x; }
    unsigned int a = 1, b = 0;
    for (int i = 0; i < 3000; ++i) { a = (a + data[i]) % 65521; b = (b + a) % 65521; }
    unsigned int crc = 0xFFFFFFFFu;
    for (int i = 0; i < 3000; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1)));
    }
    unsigned short s1 = 0, s2 = 0;
    for (int i = 0; i < 3000; ++i) { s1 = (unsigned short)((s1 + data[i]) % 255); s2 = (unsigned short)((s2 + s1) % 255); }
    put_u64((u64)((b << 16) | a)); put_u64((u64)(crc ^ 0xFFFFFFFFu)); put_u64((u64)((s2 << 8) | s1));
    return 0;
}
