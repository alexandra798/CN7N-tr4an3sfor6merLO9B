#include "fe_ofi.hpp"

static inline float mid(const L2Frame& f) {
  return 0.5f * (f.bid_p[0] + f.ask_p[0]);
}

float compute_ofi(const L2Frame& prev, const L2Frame& cur) {
  const float prev_bp = prev.bid_p[0], prev_ap = prev.ask_p[0];
  const float cur_bp  = cur.bid_p[0],  cur_ap  = cur.ask_p[0];
  const float prev_bs = prev.bid_s[0], prev_as = prev.ask_s[0];
  const float cur_bs  = cur.bid_s[0],  cur_as  = cur.ask_s[0];

  float ofi_bid = 0.0f;
  if (cur_bp > prev_bp) ofi_bid = cur_bs;
  else if (cur_bp == prev_bp) ofi_bid = cur_bs - prev_bs;
  else ofi_bid = -prev_bs;

  float ofi_ask = 0.0f;
  if (cur_ap < prev_ap) ofi_ask = prev_as;
  else if (cur_ap == prev_ap) ofi_ask = prev_as - cur_as;
  else ofi_ask = -cur_as;

  return ofi_bid + ofi_ask;
}

std::vector<float> make_feature_vector(const L2Frame& prev,
                                       const L2Frame& cur,
                                       float cum_vol_prev) {
  const float ofi = compute_ofi(prev, cur);
  const float spread = cur.ask_p[0] - cur.bid_p[0];
  const float m = mid(cur);
  const float dm = m - mid(prev);
  const float cum_vol = cum_vol_prev + cur.trade_vol;

  std::vector<float> out;
  out.resize(20 * 7);
  for (int l = 0; l < 20; ++l) {
    const int base = l * 7;
    out[base + 0] = cur.bid_s[l];
    out[base + 1] = cur.ask_s[l];
    out[base + 2] = ofi;
    out[base + 3] = spread;
    out[base + 4] = m;
    out[base + 5] = dm;
    out[base + 6] = cum_vol;
  }
  return out;
}
