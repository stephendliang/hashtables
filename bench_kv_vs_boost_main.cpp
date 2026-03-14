/*
 * bench_kv_vs_boost_main.cpp — C++ driver: boost::unordered_flat_map
 * benchmark + orchestration for sentinel/bitstealing/boost comparison.
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -c bench_kv_vs_boost.c
 *   c++ -O3 -march=native -std=c++17 -c bench_kv_vs_boost_main.cpp
 *   c++ -O3 -o bench_kv_vs_boost bench_kv_vs_boost.o bench_kv_vs_boost_main.o
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <array>

#include <boost/unordered/unordered_flat_map.hpp>

/* ================================================================
 * Shared types (must match declarations in .c)
 * ================================================================ */

#define KW 2
#define VW 1
#define OP_GET    0
#define OP_INSERT 1
#define OP_DELETE 2

extern "C" {

struct kv_work {
    uint64_t *keys;
    uint64_t *vals;
    uint32_t *op_idx;
    uint8_t  *op_type;
    uint64_t  pool_size;
    uint64_t  n_init;
    uint64_t  n_ops;
};

struct kv_result {
    double ns_per_op;
    uint64_t final_count;
};

void           kv_seed_rng(void);
struct kv_work kv_gen_insert(uint64_t n);
struct kv_work kv_gen_mixed(uint64_t n_init, uint64_t n_ops,
                            int pct_r, int pct_i, int pct_d);
void           kv_free_work(struct kv_work *w);
struct kv_result bench_sent(const struct kv_work *w, int pf);
struct kv_result bench_bs(const struct kv_work *w, int pf);

} // extern "C"

/* ================================================================
 * Timing
 * ================================================================ */

static inline double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * boost::unordered_flat_map benchmark
 * ================================================================ */

using kv_key_t = std::array<uint64_t, 2>;

static kv_result bench_boost_insert(const kv_work *w) {
    boost::unordered_flat_map<kv_key_t, uint64_t> map;
    map.reserve(w->n_ops);

    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        uint32_t idx = w->op_idx[i];
        kv_key_t k = {w->keys[idx * KW], w->keys[idx * KW + 1]};
        map.emplace(k, w->vals[idx * VW]);
    }
    double elapsed = now_sec() - t0;
    kv_result r;
    r.ns_per_op = elapsed * 1e9 / (double)w->n_ops;
    r.final_count = (uint64_t)map.size();
    return r;
}

static kv_result bench_boost_mixed(const kv_work *w) {
    boost::unordered_flat_map<kv_key_t, uint64_t> map;
    map.reserve(w->n_init + w->n_ops);

    /* pre-insert n_init keys (untimed) */
    for (uint64_t i = 0; i < w->n_init; i++) {
        kv_key_t k = {w->keys[i * KW], w->keys[i * KW + 1]};
        map.emplace(k, w->vals[i * VW]);
    }

    /* timed mixed phase */
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        uint32_t idx = w->op_idx[i];
        kv_key_t k = {w->keys[idx * KW], w->keys[idx * KW + 1]};
        switch (w->op_type[i]) {
        case OP_GET:    map.find(k); break;
        case OP_INSERT: map.emplace(k, w->vals[idx * VW]); break;
        case OP_DELETE: map.erase(k); break;
        }
    }
    double elapsed = now_sec() - t0;
    kv_result r;
    r.ns_per_op = elapsed * 1e9 / (double)w->n_ops;
    r.final_count = (uint64_t)map.size();
    return r;
}

/* ================================================================
 * Orchestration
 * ================================================================ */

static const char *g_map_filter = nullptr;

static void run(const char *name, const kv_work *w) {
    if (!g_map_filter || strcmp(g_map_filter, "sentinel") == 0) {
        kv_result r = bench_sent(w, 24);
        printf("sentinel\t%s\t%.1f\t%lu\n",
               name, r.ns_per_op, (unsigned long)r.final_count);
    }
    if (!g_map_filter || strcmp(g_map_filter, "bitstealing") == 0) {
        kv_result r = bench_bs(w, 24);
        printf("bitstealing\t%s\t%.1f\t%lu\n",
               name, r.ns_per_op, (unsigned long)r.final_count);
    }
    if (!g_map_filter || strcmp(g_map_filter, "boost") == 0) {
        kv_result r = (w->n_init == 0) ? bench_boost_insert(w)
                                       : bench_boost_mixed(w);
        printf("boost\t%s\t%.1f\t%lu\n",
               name, r.ns_per_op, (unsigned long)r.final_count);
    }
    fflush(stdout);
}

static bool should_run(const char *filter, const char *name) {
    return !filter || strcmp(filter, name) == 0;
}

int main(int argc, char **argv) {
    const char *filter = (argc > 1) ? argv[1] : nullptr;
    g_map_filter = (argc > 2) ? argv[2] : nullptr;

    kv_seed_rng();
    printf("# map\tworkload\tns_per_op\tfinal_count\n");

    /* Insert-only: 2M unique keys */
    if (should_run(filter, "insert_only")) {
        kv_work w = kv_gen_insert(2000000);
        run("insert_only", &w);
        kv_free_work(&w);
    }

    /* Insert-and-read: 500K init, 2M mixed ops (5 proportions) */
    {
        int rw[][2] = {{95, 5}, {75, 25}, {50, 50}, {25, 75}, {5, 95}};
        for (int p = 0; p < 5; p++) {
            int pi = rw[p][0], pr = rw[p][1];
            char name[64];
            snprintf(name, 64, "rw_%d_%d", pi, pr);
            if (!should_run(filter, name)) continue;
            kv_work w = kv_gen_mixed(500000, 2000000, pr, pi, 0);
            run(name, &w);
            kv_free_work(&w);
        }
    }

    /* Churn: 1M init, 2M mixed ops (4 proportions) */
    {
        int ch[][3] = {{80, 10, 10}, {50, 25, 25}, {33, 33, 34}, {20, 40, 40}};
        for (int c = 0; c < 4; c++) {
            int r = ch[c][0], ins = ch[c][1], d = ch[c][2];
            char name[64];
            snprintf(name, 64, "churn_%d_%d_%d", r, ins, d);
            if (!should_run(filter, name)) continue;
            kv_work w = kv_gen_mixed(1000000, 2000000, r, ins, d);
            run(name, &w);
            kv_free_work(&w);
        }
    }

    return 0;
}
