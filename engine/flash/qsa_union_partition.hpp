#pragma once

#include <stdexcept>

namespace chlorine_flash {

struct QsaQueryRange {
  int first;
  int count;
};

struct QsaUnionPartition {
  QsaQueryRange head;
  QsaQueryRange groups;  // query count, a multiple of four
  QsaQueryRange tail;
};

// The union list uses uint16 block ids and reserves 0xffff for padding.
// Tokens in that final block must use the per-query attention kernel.
// Every returned range lies inside [first, end), including batches shorter
// than the distance to the next aligned group. In that case one fallback
// covers the whole batch; independently rounding both edges would overlap
// them and can place the tail before the query/selection buffers.
constexpr QsaUnionPartition qsa_union_partition(int first, int end) {
  if (first < 0 || end < first || end > 262144)
    throw std::invalid_argument("QSA union query interval outside native context");
  const int ga = (first + 3) & ~3;
  const int rounded_end = end & ~3;
  const int ge = rounded_end < 65535 * 4 ? rounded_end : 65535 * 4;
  if (ga >= ge)
    return {{first, end - first}, {end, 0}, {end, 0}};
  return {{first, ga - first}, {ga, ge - ga}, {ge, end - ge}};
}

}  // namespace chlorine_flash
