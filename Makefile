CC       = cc
CXX      = c++
CFLAGS   = -O3 -march=native -std=gnu11 -Iinclude
CXXFLAGS = -O3 -march=native -std=c++17
SCALAR   = -mno-avx2 -mno-avx512f
ACFLAGS  = -O3 -march=native -std=gnu11 -Iarchive
BUILD    = build

HDRS         = $(wildcard include/*.h)
ARCHIVE_HDRS = $(wildcard archive/*.h)

# ---------------------------------------------------------------------------
# Target lists
# ---------------------------------------------------------------------------

TESTS_NATIVE = test_set_generic test_map_generic test_map64 test_map48 \
               test_set48_direct test_map48_direct test_set48_arch

TESTS_SCALAR = test_set_generic_scalar test_map_generic_scalar \
               test_map64_scalar test_set48_direct_scalar \
               test_map48_direct_scalar test_set48_arch_scalar

TESTS = $(TESTS_NATIVE) $(TESTS_SCALAR)

BENCHES = bench_map64_layout bench_map_layout bench_map_pf_tuning \
          bench_map48 bench_map48_direct bench_map48_arch \
          bench_tcp_state bench_tcp_pareto bench_map64_mixed \
          bench_512 bench_avx2 bench_map_vs_boost

ARCHIVE_TARGETS = test_map128 bench_map128_throughput bench_map128_throughput_bs \
                  bench_map128_delete bench_map128_delete_pf

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------

.PHONY: all test bench archive clean $(TESTS) $(BENCHES) $(ARCHIVE_TARGETS)

all: $(addprefix $(BUILD)/,$(TESTS))

test: all
	@pass=0; \
	for t in $(TESTS); do \
		printf '  RUN   %s\n' "$$t"; \
		if $(BUILD)/$$t; then \
			pass=$$((pass + 1)); \
		else \
			printf '  FAIL  %s\n' "$$t"; \
			exit 1; \
		fi; \
	done; \
	printf '  OK    %d/%d tests passed\n' "$$pass" "$$pass"

bench: $(addprefix $(BUILD)/,$(BENCHES))

archive: $(addprefix $(BUILD)/,$(ARCHIVE_TARGETS))

clean:
	rm -rf $(BUILD)

# ---------------------------------------------------------------------------
# Convenience aliases (make test_map64 → build/test_map64)
# ---------------------------------------------------------------------------

$(TESTS) $(BENCHES): %: $(BUILD)/%
$(ARCHIVE_TARGETS): %: $(BUILD)/%

# ---------------------------------------------------------------------------
# Build directory (order-only prerequisite)
# ---------------------------------------------------------------------------

$(BUILD):
	@mkdir -p $@

# ---------------------------------------------------------------------------
# Pattern rules — binary name matches source basename
# ---------------------------------------------------------------------------

# test/test_FOO.c → build/test_FOO
$(BUILD)/test_%: test/test_%.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) -o $@ $<

# bench/bench_FOO.c → build/bench_FOO
$(BUILD)/bench_%: bench/bench_%.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# Explicit rules — non-matching source names
# ---------------------------------------------------------------------------

$(BUILD)/bench_512: bench/bench_map64_backends.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD)/bench_avx2: bench/bench_map64_backends.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) -mno-avx512f -o $@ $< -lm

# ---------------------------------------------------------------------------
# Scalar test variants
# ---------------------------------------------------------------------------

$(BUILD)/test_set_generic_scalar: test/test_set_generic.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

$(BUILD)/test_map_generic_scalar: test/test_map_generic.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

$(BUILD)/test_map64_scalar: test/test_map64.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

$(BUILD)/test_set48_direct_scalar: test/test_set48_direct.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

$(BUILD)/test_map48_direct_scalar: test/test_map48_direct.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

$(BUILD)/test_set48_arch_scalar: test/test_set48_arch.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) $(SCALAR) -o $@ $<

# ---------------------------------------------------------------------------
# Multi-file benchmark (C + C++ linkage, requires boost headers)
# ---------------------------------------------------------------------------

$(BUILD)/bench_map_vs_boost: $(BUILD)/bench_map_vs_boost.o $(BUILD)/bench_map_vs_boost_main.o | $(BUILD)
	@printf '  LD    $@\n'
	@$(CXX) -O3 -o $@ $^

$(BUILD)/bench_map_vs_boost.o: bench/bench_map_vs_boost.c $(HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/bench_map_vs_boost_main.o: bench/bench_map_vs_boost_main.cpp | $(BUILD)
	@printf '  CXX   $@\n'
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

# ---------------------------------------------------------------------------
# Archive targets (opt-in: make archive)
# ---------------------------------------------------------------------------

$(BUILD)/test_map128: archive/test_map128.c $(ARCHIVE_HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(ACFLAGS) -o $@ $<

$(BUILD)/bench_map128_throughput: archive/bench_map128_throughput.c $(ARCHIVE_HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(ACFLAGS) -o $@ $<

$(BUILD)/bench_map128_throughput_bs: archive/bench_map128_throughput.c $(ARCHIVE_HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(ACFLAGS) -DBITSTEALING -o $@ $<

$(BUILD)/bench_map128_delete: archive/bench_map128_delete.c $(ARCHIVE_HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(ACFLAGS) -o $@ $<

$(BUILD)/bench_map128_delete_pf: archive/bench_map128_delete_pf.c $(ARCHIVE_HDRS) | $(BUILD)
	@printf '  CC    $@\n'
	@$(CC) $(ACFLAGS) -o $@ $<
