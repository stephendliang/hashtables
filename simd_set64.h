/* simd_set64.h — uint64_t set, thin wrapper over simd_map64.h (VW=0)

Includes simd_map64.h in set mode. Keys stored directly in 8-wide groups
(one cache line). SIMD compares all 8 at once — zero false positives,
no metadata, no scalar verify. Key=0 reserved as empty sentinel.

API: _init, _destroy, _init_cap, _insert, _insert_unique, _contains,
     _delete, _op, _prefetch, _prefetch_insert, _prefetch2. */
#pragma once

#define SIMD_MAP_NAME simd_set64
#include "simd_map64.h"
