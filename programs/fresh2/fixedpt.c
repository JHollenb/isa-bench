#include "crt.h"
// Q16.16 fixed-point math: integer square root by Newton steps, sine/cosine by CORDIC rotation, and a damped
// oscillator integrated with the semi-implicit Euler rule.
typedef long fx;
#define ONE 65536L
static fx fmul(fx a, fx b) { return (a * b) >> 16; }
static fx fdiv(fx a, fx b) { return (a << 16) / b; }
static fx fsqrt(fx a) {
    if (a <= 0) return 0;
    fx x = a > ONE ? a / 2 : ONE;
    for (int i = 0; i < 20; ++i) x = (x + fdiv(a, x)) / 2;
    return x;
}
static const fx atan_tab[16] = {51472, 30386, 16055, 8150, 4091, 2047, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2};
static void cordic(fx angle, fx *c, fx *s) {
    fx x = 39797, y = 0, z = angle;  // x = K * ONE (K = 0.607253)
    for (int i = 0; i < 16; ++i) {
        fx dx = x >> i, dy = y >> i;
        if (z >= 0) { x -= dy; y += dx; z -= atan_tab[i]; }
        else { x += dy; y -= dx; z += atan_tab[i]; }
    }
    *c = x; *s = y;
}
int cmain(void) {
    u64 acc = 0;
    for (fx v = ONE / 4; v < 400 * ONE; v += 3 * ONE + 12345) acc = acc * 7 + (u64)fsqrt(v);
    fx sc = 0;
    for (fx a = -102943; a <= 102943; a += 997) {  // about -pi/2 .. pi/2
        fx c, s;
        cordic(a, &c, &s);
        sc += fmul(c, c) + fmul(s, s) - ONE;
        acc ^= (u64)(s + 3 * c);
    }
    fx pos = ONE, vel = 0, k = ONE / 3, damp = ONE / 50, dt = ONE / 16;
    for (int i = 0; i < 3000; ++i) {
        vel -= fmul(fmul(k, pos) + fmul(damp, vel), dt);
        pos += fmul(vel, dt);
    }
    put_u64(acc); put_u64((u64)sc); put_u64((u64)pos); put_u64((u64)vel);
    return 0;
}
