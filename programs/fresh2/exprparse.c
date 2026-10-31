#include "crt.h"
// Recursive-descent parser/evaluator for integer expressions (+ - * / % unary minus, parentheses) over a fixed set of
// expression strings; each is evaluated many times with a variable x substituted.
static const char *src;
static long xval;
static long expr(void);
static void skip(void) { while (*src == ' ') ++src; }
static long atom(void) {
    skip();
    if (*src == '(') { ++src; long v = expr(); skip(); if (*src == ')') ++src; return v; }
    if (*src == '-') { ++src; return -atom(); }
    if (*src == 'x') { ++src; return xval; }
    long v = 0;
    while (*src >= '0' && *src <= '9') v = v * 10 + (*src++ - '0');
    return v;
}
static long term(void) {
    long v = atom();
    for (;;) {
        skip();
        char op = *src;
        if (op != '*' && op != '/' && op != '%') return v;
        ++src;
        long r = atom();
        if (op == '*') v *= r;
        else if (r != 0) v = op == '/' ? v / r : v % r;
    }
}
static long expr(void) {
    long v = term();
    for (;;) {
        skip();
        char op = *src;
        if (op != '+' && op != '-') return v;
        ++src;
        long r = term();
        v = op == '+' ? v + r : v - r;
    }
}
static const char *exprs[] = {"1 + 2 * 3", "(x + 7) * (x - 3) % 1000", "-(x * x) / 7 + 42", "((x % 13) * 17 - (x / 3)) * 2",
                              "100 - x * (2 + x % 5)", "(1+(2+(3+(4+x))))*-1"};
int cmain(void) {
    long acc = 0;
    u64 mix = 0;
    for (long x = -60; x < 60; ++x) {
        xval = x;
        for (int i = 0; i < 6; ++i) {
            src = exprs[i];
            long v = expr();
            acc += v;
            mix = mix * 1099511628211ull ^ (u64)v;
        }
    }
    put_u64((u64)acc); put_u64(mix);
    return 0;
}
