#include "preproc.hpp"
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

// Extremely small JSON parser for {"mean":[...], "std":[...]} with floats.
// Only supports this exact structure.
static bool parse_float_array(const std::string& s, const std::string& key, std::vector<float>& out) {
  auto pos = s.find(key);
  if (pos == std::string::npos) return false;
  pos = s.find('[', pos);
  auto end = s.find(']', pos);
  if (pos == std::string::npos || end == std::string::npos) return false;
  std::string body = s.substr(pos + 1, end - pos - 1);
  std::stringstream ss(body);
  out.clear();
  while (ss.good()) {
    std::string tok;
    if (!std::getline(ss, tok, ',')) break;
    // trim
    tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(), [](unsigned char ch){ return !std::isspace(ch); }));
    tok.erase(std::find_if(tok.rbegin(), tok.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), tok.end());
    if (tok.empty()) continue;
    out.push_back(std::stof(tok));
  }
  return !out.empty();
}

bool load_norm_stats_json(const std::string& path, NormStats& out) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::stringstream buf;
  buf << f.rdbuf();
  std::string s = buf.str();
  std::vector<float> mean, st;
  //if (!parse_float_array(s, ""mean"", mean)) return false;
  //if (!parse_float_array(s, ""std"", st)) return false;
  if (!parse_float_array(s, "\"mean\"", mean)) return false;
  if (!parse_float_array(s, "\"std\"", st)) return false;


  out.mean = std::move(mean);
  out.stdev = std::move(st);
  for (auto& v : out.stdev) if (v < 1e-8f) v = 1.0f;
  return true;
}

WindowAssembler::WindowAssembler(int T, int L, int C) : T_(T), L_(L), C_(C) {
  ring_.assign(static_cast<size_t>(T_) * L_ * C_, 0.0f);
  scratch_.assign(static_cast<size_t>(T_) * L_ * C_, 0.0f);
}

void WindowAssembler::push_frame(const std::vector<float>& lc) {
  if ((int)lc.size() != L_ * C_) return;
  const size_t offset = static_cast<size_t>(idx_) * L_ * C_;
  std::copy(lc.begin(), lc.end(), ring_.begin() + offset);
  idx_ = (idx_ + 1) % T_;
  if (idx_ == 0) full_ = true;
}

void WindowAssembler::normalize_last(const NormStats& stats) {
  if ((int)stats.mean.size() != C_ || (int)stats.stdev.size() != C_) return;
  int last = idx_ - 1;
  if (last < 0) last += T_;
  size_t offset = static_cast<size_t>(last) * L_ * C_;
  for (int l = 0; l < L_; ++l) {
    for (int c = 0; c < C_; ++c) {
      size_t i = offset + static_cast<size_t>(l) * C_ + c;
      ring_[i] = (ring_[i] - stats.mean[c]) / stats.stdev[c];
    }
  }
}

const float* WindowAssembler::view_input_time_ordered() {
  if (!full_) return nullptr;

  // idx_ points to next write index; that index is the oldest frame.
  if (idx_ == 0) {
    // already time-ordered
    return ring_.data();
  }

  const size_t frame_size = static_cast<size_t>(L_) * C_;
  // copy [idx_..T_) then [0..idx_)
  size_t out_off = 0;
  for (int t = idx_; t < T_; ++t) {
    const size_t in_off = static_cast<size_t>(t) * frame_size;
    std::copy(ring_.begin() + in_off, ring_.begin() + in_off + frame_size, scratch_.begin() + out_off);
    out_off += frame_size;
  }
  for (int t = 0; t < idx_; ++t) {
    const size_t in_off = static_cast<size_t>(t) * frame_size;
    std::copy(ring_.begin() + in_off, ring_.begin() + in_off + frame_size, scratch_.begin() + out_off);
    out_off += frame_size;
  }
  return scratch_.data();
}
