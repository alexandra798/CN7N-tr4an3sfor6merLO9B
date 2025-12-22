#include "preproc.hpp"
#include "fe_ofi.hpp"
#include "litcvg_session.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

// Tiny YAML reader for runtime.yaml key: value lines (no nesting).
static bool load_runtime_yaml(const std::string& path,
                              std::string& model_path,
                              std::string& norm_path,
                              int& T, int& L, int& C, int& threads) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  std::string line;
  while (std::getline(f, line)) {
    auto pos = line.find(':');
    if (pos == std::string::npos) continue;
    std::string k = line.substr(0, pos);
    std::string v = line.substr(pos + 1);
    // trim spaces
    auto trim = [](std::string& s) {
      while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
      while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
      if (!s.empty() && s.front()=='"') s.erase(s.begin());
      if (!s.empty() && s.back()=='"') s.pop_back();
    };
    trim(k); trim(v);
    if (k == "model_path") model_path = v;
    else if (k == "norm_stats") norm_path = v;
    else if (k == "T") T = std::stoi(v);
    else if (k == "L") L = std::stoi(v);
    else if (k == "C_raw") C = std::stoi(v);
    else if (k == "threads") threads = std::stoi(v);
  }
  return !model_path.empty();
}

static L2Frame make_synth_frame(float mid, float spread) {
  L2Frame f;
  float tick = 0.01f;
  float bid0 = mid - spread / 2.0f;
  float ask0 = mid + spread / 2.0f;
  for (int i = 0; i < 20; ++i) {
    f.bid_p[i] = bid0 - i * tick;
    f.ask_p[i] = ask0 + i * tick;
    f.bid_s[i] = 10.0f + 0.1f * i;
    f.ask_s[i] = 10.0f + 0.1f * i;
  }
  f.trade_vol = 0.5f;
  return f;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: main_stream path/to/runtime.yaml\n";
    return 1;
  }

  std::string model_path, norm_path;
  int T = 100, L = 20, C = 7, threads = 1;
  if (!load_runtime_yaml(argv[1], model_path, norm_path, T, L, C, threads)) {
    std::cerr << "failed to load runtime.yaml\n";
    return 1;
  }

  NormStats stats;
  if (!load_norm_stats_json(norm_path, stats)) {
    std::cerr << "warning: failed to load norm stats, proceeding without normalize\n";
  }

  WindowAssembler wa(T, L, C);
  LiTCVGSession sess(model_path, threads);

  L2Frame prev = make_synth_frame(100.0f, 0.01f);
  float cum_vol = 0.0f;

  const int N = 2000;
  auto t0 = std::chrono::high_resolution_clock::now();
  int infer_cnt = 0;

  for (int i = 0; i < N; ++i) {
    float mid = 100.0f + 0.002f * i;
    L2Frame cur = make_synth_frame(mid, 0.01f);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    cum_vol += cur.trade_vol;

    wa.push_frame(feat);
    if (!stats.mean.empty()) wa.normalize_last(stats);

    const float* ptr = wa.view_input_time_ordered();
    if (ptr) {
      std::array<int64_t,4> shape{1, T, L, C};
      std::vector<float> o10, o50, o100;
      if (sess.infer(ptr, shape, o10, o50, o100)) {
        ++infer_cnt;
      }
    }
    prev = cur;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "frames=" << N << " inferences=" << infer_cnt << " total_ms=" << ms
            << " avg_ms_per_frame=" << (ms / N) << "\n";
  return 0;
}
