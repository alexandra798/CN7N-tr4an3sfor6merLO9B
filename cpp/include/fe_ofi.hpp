#pragma once
#include <array>
#include <vector>

struct L2Frame {
  std::array<float, 20> bid_p, ask_p;
  std::array<float, 20> bid_s, ask_s;
  float trade_vol = 0.0f;
};

float compute_ofi(const L2Frame& prev, const L2Frame& cur);

// Contract: returns (L*C_raw) flattened row-major by level then channel:
// for level l: [bid_size, ask_size, OFI, spread, mid, dmid, cum_vol]
std::vector<float> make_feature_vector(const L2Frame& prev,
                                       const L2Frame& cur,
                                       float cum_vol_prev);
