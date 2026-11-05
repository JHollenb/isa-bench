#include "crt.h"
// Single-source shortest paths (array Dijkstra, O(V^2)) on a generated weighted digraph of 120 vertices, from three
// sources; prints a distance checksum, the farthest vertex and the number of unreachable vertices.
#define V 120
#define INF 0x3fffffffffffffffLL
static long w[V][V];
static long dist[V];
static unsigned char done[V];
int cmain(void) {
    unsigned long s = 99991;
    for (int i = 0; i < V; ++i)
        for (int j = 0; j < V; ++j) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            long r = (long)(s >> 33);
            w[i][j] = (i != j && r % 7 < 2) ? 1 + r % 97 : -1;
        }
    u64 sum = 0, far = 0, unreach = 0;
    for (int src = 0; src < 3; ++src) {
        for (int i = 0; i < V; ++i) { dist[i] = INF; done[i] = 0; }
        dist[src * 37] = 0;
        for (int it = 0; it < V; ++it) {
            int u = -1;
            for (int i = 0; i < V; ++i) if (!done[i] && (u < 0 || dist[i] < dist[u])) u = i;
            if (dist[u] == INF) break;
            done[u] = 1;
            for (int v = 0; v < V; ++v)
                if (w[u][v] > 0 && dist[u] + w[u][v] < dist[v]) dist[v] = dist[u] + w[u][v];
        }
        long best = -1;
        for (int i = 0; i < V; ++i) {
            if (dist[i] == INF) { ++unreach; continue; }
            sum = sum * 1000003 + (u64)dist[i];
            if (dist[i] > best) { best = dist[i]; far = (u64)i; }
        }
    }
    put_u64(sum); put_u64(far); put_u64(unreach);
    return 0;
}
