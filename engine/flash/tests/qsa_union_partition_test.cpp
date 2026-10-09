#include "../qsa_union_partition.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static void check(int first, int end) {
  const auto p = chlorine_flash::qsa_union_partition(first, end);
  int cursor = first;
  for (const auto r : {p.head, p.groups, p.tail}) {
    require(r.count >= 0 && r.first == cursor, "query gap or overlapping launches");
    require(r.first >= first && r.first + r.count <= end, "query/selection out of bounds");
    cursor += r.count;
  }
  require(cursor == end, "missing query rows");
  if (p.groups.count) {
    require((p.groups.first & 3) == 0 && (p.groups.count & 3) == 0,
            "partial query group sent to union kernel");
    require(p.groups.first + p.groups.count <= 65535 * 4,
            "padding block id used as a real union block");
  }
}

int main() {
  // The reported continuation used to issue rows 2052..2055 for a request
  // containing only 2053 and 2054 (including a negative selection offset).
  const auto short_batch = chlorine_flash::qsa_union_partition(2053, 2055);
  require(short_batch.head.first == 2053 && short_batch.head.count == 2 &&
              short_batch.groups.count == 0 && short_batch.tail.count == 0,
          "two-row continuation regression");
  size_t cases = 0;
  for (int first = 0; first <= 2080; ++first)
    for (int count = 0; count <= 17; ++count) { check(first, first + count); ++cases; }
  for (int first : {2051, 2052, 2053, 2054, 8191, 8192, 8193, 245759})
    for (int count : {1, 2, 3, 4, 5, 8, 9, 65, 2048, 8192, 16384}) {
      check(first, first + count); ++cases;
    }
  // Exercise every alignment in the sentinel block and its neighbours.
  for (int first = 262120; first <= 262144; ++first)
    for (int end = first; end <= 262144; ++end) { check(first, end); ++cases; }
  check(2051, 262144);
  for (const auto bad : {std::pair<int, int>{-1, 2}, {4, 3}, {0, 262145}}) {
    bool rejected = false;
    try { (void)chlorine_flash::qsa_union_partition(bad.first, bad.second); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid query interval accepted");
  }
  std::cout << "PASS QSA union partitions: " << cases + 1
            << " intervals, exact coverage and no out-of-range selection offsets\n";
}
