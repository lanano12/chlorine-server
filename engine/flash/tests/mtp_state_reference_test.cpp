#include "mtp_state_reference.hpp"

#include <iostream>
#include <stdexcept>

using namespace mtp_reference;

static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static Round round(int pos, int current, std::vector<int> drafts,
                   std::vector<int> verified) {
  return {pos, pos - 1, current, std::move(drafts), std::move(verified)};
}

int main() {
  // First-draft rejection: correction is emitted, only `current` is consumed.
  auto first = greedy_round(round(9, 40, {11, 12, 13}, {99, 12, 13, 14}), 12);
  require(first.accepted == 0 && first.commit_depth == 0, "first rejection depth");
  require(first.emitted == std::vector<int>{99}, "first rejection output");
  require(first.next_trunk_pos == 10 && first.next_mtp_pos == 9 &&
              first.next_current == 99,
          "first rejection positions/current");

  // Interior rejection: keep the accepted prefix, then the target correction.
  auto middle = greedy_round(round(10, 99, {21, 22, 23}, {21, 88, 23, 24}), 12);
  require(middle.accepted == 1 && middle.commit_depth == 1,
          "interior rejection depth");
  require(middle.emitted == std::vector<int>({21, 88}), "interior output");
  require(middle.next_trunk_pos == 12 && middle.next_mtp_pos == 11 &&
              middle.next_current == 88,
          "interior rollback positions/current");

  // Full acceptance commits all drafts and emits the bonus target token.
  auto full = greedy_round(round(12, 88, {31, 32}, {31, 32, 33}), 12);
  require(full.accepted == 2 && full.commit_depth == 2, "full acceptance depth");
  require(full.emitted == std::vector<int>({31, 32, 33}), "full acceptance output");
  require(full.next_trunk_pos == 15 && full.next_mtp_pos == 14 &&
              full.next_current == 33,
          "full acceptance positions/current");

  // A truncated output budget must not ingest the hidden correction/bonus.
  auto tail = greedy_round(round(20, 7, {41, 42, 43}, {41, 42, 43, 44}), 2);
  require(tail.accepted == 3 && tail.commit_depth == 2,
          "budget truncation commit depth");
  require(tail.emitted == std::vector<int>({41, 42}), "budget truncation output");
  require(tail.next_trunk_pos == 23 && tail.next_mtp_pos == 22 &&
              tail.next_current == 43,
          "budget truncation state");

  bool rejected = false;
  try {
    (void)greedy_round(Round{5, 3, 1, {2}, {3, 4}}, 2);  // mismatched lockstep
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "out-of-lockstep MTP state accepted");
  rejected = false;
  try {
    (void)greedy_round(round(5, 1, {2}, {3}), 2);  // missing target row
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "malformed verify rows accepted");

  std::cout << "PASS MTP greedy proposal alignment, acceptance, rollback positions, and budget tail\n";
}
