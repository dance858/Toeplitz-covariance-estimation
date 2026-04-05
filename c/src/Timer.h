#ifndef NML_TIMER_H
#define NML_TIMER_H

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
typedef struct
{
    LARGE_INTEGER start, end;
} Timer;
static inline void clock_gettime_monotonic(Timer *timer, int is_start)
{
    if (is_start)
    {
        QueryPerformanceCounter(&timer->start);
    }
    else
    {
        QueryPerformanceCounter(&timer->end);
    }
}
static inline double get_elapsed_seconds(const Timer *timer)
{
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return (double) (timer->end.QuadPart - timer->start.QuadPart) /
           (double) freq.QuadPart;
}
#define clock_gettime(CLOCK, PTR)                                                   \
    clock_gettime_monotonic((Timer *) (PTR), ((PTR) == &((Timer *) (PTR))->start))
#define GET_ELAPSED_SECONDS(timer) get_elapsed_seconds(&(timer))
#else
#include <time.h>
typedef struct
{
    struct timespec start, end;
} Timer;

#define GET_ELAPSED_SECONDS(timer)                                                  \
    ((double) ((timer).end.tv_sec - (timer).start.tv_sec) +                         \
     (double) ((timer).end.tv_nsec - (timer).start.tv_nsec) / 1e9)
#endif

#endif
