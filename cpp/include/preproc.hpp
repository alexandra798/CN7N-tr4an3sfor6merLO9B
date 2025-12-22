#pragma once
#include <vector>
#include <string>

struct NormStats {
  std::vector<float> mean;
  std::vector<float> stdev;
};

// Minimal JSON loader stub.
// For production, replace with nlohmann/json or another robust parser.
bool load_norm_stats_json(const std::string& path, NormStats& out);

class WindowAssembler {
public:
  WindowAssembler(int T, int L, int C);

  // lc: flattened (L*C) vector (level-major then channel)
  void push_frame(const std::vector<float>& lc);

  // Normalize in-place for the last pushed frame (requires mean/std of length C)
  void normalize_last(const NormStats& stats);

  // Returns pointer to contiguous (1,T,L,C) buffer in **time order** (oldest -> newest).
  // This may copy into an internal scratch buffer when ring offset != 0.
  const float* view_input_time_ordered();

  int T() const { return T_; }
  int L() const { return L_; }
  int C() const { return C_; }
  bool full() const { return full_; }

private:
  int T_, L_, C_;
  int idx_ = 0;      // next write index in [0, T)
  bool full_ = false;

  std::vector<float> ring_;    // layout: [t][l][c] with t in ring indices
  std::vector<float> scratch_; // time-ordered contiguous buffer for ORT
};
