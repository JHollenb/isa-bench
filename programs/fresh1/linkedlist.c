#include "crt.h"
// Singly linked list in a static node pool: build 300 nodes, insert sorted, delete every third, reverse, walk.
#define N 300
struct node { struct node *next; long key; int tag; };
static struct node pool[N];
static struct node *head;
static void insert_sorted(struct node *n) {
    struct node **p = &head;
    while (*p && (*p)->key < n->key) p = &(*p)->next;
    n->next = *p;
    *p = n;
}
static void drop_every(int k) {
    struct node **p = &head;
    int i = 0;
    while (*p) {
        if (++i % k == 0) *p = (*p)->next;
        else p = &(*p)->next;
    }
}
static void reverse(void) {
    struct node *prev = 0, *cur = head;
    while (cur) { struct node *nx = cur->next; cur->next = prev; prev = cur; cur = nx; }
    head = prev;
}
int cmain(void) {
    unsigned x = 12345;
    for (int i = 0; i < N; ++i) {
        x = x * 1103515245u + 12345u;
        pool[i].key = (long)(x >> 8) - 4000000;
        pool[i].tag = i;
        insert_sorted(&pool[i]);
    }
    drop_every(3);
    reverse();
    long count = 0, wsum = 0, desc = 1, prevk = 0x7fffffffffffffffL;
    for (struct node *n = head; n; n = n->next) {
        ++count;
        wsum += n->key * (long)(n->tag + 1);
        if (n->key > prevk) desc = 0;
        prevk = n->key;
    }
    put_u64((u64)count); put_u64((u64)wsum); put_u64((u64)desc);
    return 0;
}
