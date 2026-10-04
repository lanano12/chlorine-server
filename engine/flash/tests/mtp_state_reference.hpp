#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

// Host-only oracle for Flash's MTP stream-position and greedy commit contract.
// It deliberately does not model MTP tensor arithmetic or device rollback.
namespace mtp_reference {

struct Round {
  int trunk_pos;  // next trunk input position
  int mtp_pos;    // next MTP KV position; invariant mtp_pos == trunk_pos - 1
  int current;    // already selected token for trunk_pos
  std::vector<int> drafts;
  std::vector<int> verified;  // row j predicts verified[j]
};

struct Result {
  int accepted;
  int commit_depth;
  int next_trunk_pos;
  int next_mtp_pos;
  int next_current;
  std::vector<int> emitted;  // excludes `current`, which was emitted earlier
};

inline Result greedy_round(const Round& r, int remaining) {
  if (r.trunk_pos < 1 || r.mtp_pos != r.trunk_pos - 1)
    throw std::invalid_argument("MTP/trunk positions are not in lockstep");
  if (remaining < 1 || r.drafts.empty() || r.drafts.size() > 8 ||
      r.verified.size() != r.drafts.size() + 1)
    throw std::invalid_argument("invalid speculative round shape");

  int accepted = 0;
  while (accepted < (int)r.drafts.size() &&
         r.drafts[accepted] == r.verified[accepted])
    ++accepted;

  // Commit accepted drafts and the target correction/bonus, unless the output
  // budget ends first. No token beyond remaining may enter either state.
  int commit_depth = accepted;
  int output_count = accepted + 1;
  if (output_count > remaining) {
    output_count = remaining;
    commit_depth = std::min(accepted, remaining);
  }
  std::vector<int> emitted;
  emitted.reserve(output_count);
  for (int i = 0; i < std::min(commit_depth, output_count); ++i)
    emitted.push_back(r.drafts[i]);
  if ((int)emitted.size() < output_count)
    emitted.push_back(r.verified[accepted]);

  // `current` is always consumed. Each committed draft is consumed too.
  const int next_pos = r.trunk_pos + 1 + commit_depth;
  const int next_mtp = next_pos - 1;
  const int next_current = r.verified[commit_depth];
  return {accepted, commit_depth, next_pos, next_mtp, next_current,
          std::move(emitted)};
}

}  // namespace mtp_reference
