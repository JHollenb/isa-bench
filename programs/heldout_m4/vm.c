#include "crt.h"
// A small stack bytecode VM (switch dispatch) running a loop that sums i*i mod 7 for i < 2000 and counts primes < 400.
enum { PUSH, LOAD, STORE, ADD, SUB, MUL, MOD, LT, JZ, JMP, DUP, POP, HALT };
static long code[] = {
    PUSH, 0, STORE, 0,            /* i = 0 */
    PUSH, 0, STORE, 1,            /* s = 0 */
    LOAD, 0, PUSH, 2000, LT, JZ, 37,
    LOAD, 1, LOAD, 0, LOAD, 0, MUL, PUSH, 7, MOD, ADD, STORE, 1,
    LOAD, 0, PUSH, 1, ADD, STORE, 0, JMP, 8,
};
static long run(const long *pc0, long *vars) {
    long stack[64], sp = 0;
    const long *pc = pc0;
    for (;;) {
        long op = *pc++;
        switch (op) {
        case PUSH: stack[sp++] = *pc++; break;
        case LOAD: stack[sp++] = vars[*pc++]; break;
        case STORE: vars[*pc++] = stack[--sp]; break;
        case ADD: --sp; stack[sp - 1] += stack[sp]; break;
        case SUB: --sp; stack[sp - 1] -= stack[sp]; break;
        case MUL: --sp; stack[sp - 1] *= stack[sp]; break;
        case MOD: --sp; stack[sp - 1] %= stack[sp]; break;
        case LT: --sp; stack[sp - 1] = stack[sp - 1] < stack[sp]; break;
        case JZ: { long t = *pc++; if (!stack[--sp]) pc = pc0 + t; break; }
        case JMP: pc = pc0 + *pc; break;
        case DUP: stack[sp] = stack[sp - 1]; ++sp; break;
        case POP: --sp; break;
        default: return sp;
        }
        if (pc - pc0 >= (long)(sizeof(code) / sizeof(code[0]))) return sp;
    }
}
int cmain(void) {
    long vars[4] = {0, 0, 0, 0};
    run(code, vars);
    u64 primes = 0;
    for (long n = 2; n < 400; ++n) { long p = 1; for (long d = 2; d * d <= n; ++d) if (n % d == 0) { p = 0; break; } primes += p; }
    put_u64((u64)vars[1]); put_u64((u64)vars[0]); put_u64(primes);
    return 0;
}
