#include "crt.h"
// Calendar arithmetic: days from civil date (proleptic Gregorian), weekday, leap years, and date from day number,
// round-tripped over a stride of dates; counts Friday-the-13ths 1990..2030.
static long days_from_civil(long y, unsigned m, unsigned d) {
    y -= m <= 2;
    long era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long)doe - 719468;
}
static void civil_from_days(long z, long *y, unsigned *m, unsigned *d) {
    z += 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long yy = (long)yoe + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = yy + (*m <= 2);
}
static int leap(long y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
int cmain(void) {
    u64 fri13 = 0, leaps = 0, ok = 1, h = 0;
    for (long y = 1990; y <= 2030; ++y) {
        leaps += (u64)leap(y);
        for (unsigned m = 1; m <= 12; ++m) {
            long z = days_from_civil(y, m, 13);
            long wd = (z + 4) % 7;  // 1970-01-01 was a Thursday (4)
            if (wd < 0) wd += 7;
            fri13 += wd == 5;
        }
    }
    for (long z = -800000; z < 800000; z += 7919) {
        long y; unsigned m, d;
        civil_from_days(z, &y, &m, &d);
        ok &= (u64)(days_from_civil(y, m, d) == z);
        h = h * 1315423911u + (u64)(y * 372 + m * 31 + d);
    }
    put_u64(fri13); put_u64(leaps); put_u64(ok); put_u64(h);
    return 0;
}
