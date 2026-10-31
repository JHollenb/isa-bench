#include "crt.h"
// Single-producer/single-consumer ring buffer (power-of-two capacity, free-running head/tail counters) driven by a
// bursty schedule; checks FIFO order and checksums the consumed stream.
#define CAP 64
static unsigned buf[CAP];
static unsigned head, tail;
static int push(unsigned v) { if (head - tail == CAP) return 0; buf[head++ & (CAP - 1)] = v; return 1; }
static int pop(unsigned *v) { if (head == tail) return 0; *v = buf[tail++ & (CAP - 1)]; return 1; }
int cmain(void) {
    unsigned next_in = 0, next_out = 0, s = 0x1234;
    u64 sum = 0, dropped = 0, order_ok = 1, maxfill = 0;
    for (int round = 0; round < 3000; ++round) {
        s = s * 1664525u + 1013904223u;
        int burst = (int)(s >> 28);
        for (int k = 0; k < burst; ++k) {
            if (push(next_in * 2654435761u)) ++next_in;
            else ++dropped;
        }
        if (head - tail > maxfill) maxfill = head - tail;
        int take = (int)((s >> 20) & 7);
        unsigned v;
        for (int k = 0; k < take && pop(&v); ++k) {
            order_ok &= (u64)(v == next_out * 2654435761u);
            ++next_out;
            sum = (sum << 5 | sum >> 59) ^ v;
        }
    }
    put_u64(sum); put_u64(dropped); put_u64(order_ok); put_u64(maxfill);
    return 0;
}
