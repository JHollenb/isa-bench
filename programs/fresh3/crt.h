// Minimal freestanding runtime for the opcode-mix programs: _start, write, exit, decimal print.
typedef unsigned long u64;
__asm__(".globl _start\n_start:\n xor %rbp, %rbp\n and $-16, %rsp\n call cmain\n mov %eax, %edi\n mov $60, %eax\n syscall\n hlt\n");
static long sys3(long n, long a, long b, long c) {
    long r;
    __asm__ volatile("syscall" : "=a"(r) : "a"(n), "D"(a), "S"(b), "d"(c) : "rcx", "r11", "memory");
    return r;
}
static void put(const char *s, long n) { sys3(1, 1, (long)s, n); }
static void put_u64(u64 v) {
    char buf[24];
    int i = 23;
    buf[i] = '\n';
    do { buf[--i] = (char)('0' + v % 10); v /= 10; } while (v);
    put(buf + i, 24 - i);
}
void *memcpy(void *d, const void *s, u64 n) { char *a = d; const char *b = s; while (n--) *a++ = *b++; return d; }
void *memset(void *d, int c, u64 n) { char *a = d; while (n--) *a++ = (char)c; return d; }
