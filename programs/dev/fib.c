#include "crt.h"
static u64 fib(u64 n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
int cmain(void) { put_u64(fib(20)); return 0; }
