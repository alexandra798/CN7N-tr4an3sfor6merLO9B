#include "perf_stats.hpp"
#include "litcvg_session.hpp"
#include "preproc.hpp"
#include "fe_ofi.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

// ===== Configuration =====
struct BenchConfig {
  std::string model_path = "artifacts/model.onnx";
  std::string norm_path = "artifacts/norm_stats.json";
  int T = 100;
  int L = 20;
  int C = 7;
  int warmup = 100;
  int iterations = 1000;
  int threads = 1;
};

// ===== Synthetic data generator =====
static L2Frame make_test_frame(int idx) {
  L2Frame f;
  float mid = 100.0f + 0.001f * idx;
  float spread = 0.01f;
  float bid0 = mid - spread / 2.0f;
  float ask0 = mid + spread / 2.0f;
  float tick = 0.01f;
  
  for (int i = 0; i < 20; ++i) {
    f.bid_p[i] = bid0 - i * tick;
    f.ask_p[i] = ask0 + i * tick;
    f.bid_s[i] = 10.0f + 0.1f * i;
    f.ask_s[i] = 10.0f + 0.1f * i;
  }
  f.trade_vol = 0.5f;
  
  return f;
}

// ===== Test 1: Feature Extraction Latency =====
void benchmark_feature_extraction(const BenchConfig& cfg) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 1: Feature Extraction Latency\n";
  std::cout << std::string(60, '=') << "\n";
  
  PerfStats stats("feature_extraction");
  
  L2Frame prev = make_test_frame(0);
  L2Frame cur;
  float cum_vol = 0.0f;
  
  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    cur = make_test_frame(i);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  // Benchmark
  for (int i = 0; i < cfg.iterations; ++i) {
    cur = make_test_frame(i + cfg.warmup);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    auto feat = make_feature_vector(prev, cur, cum_vol);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.record_latency(latency_ns);
    
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  stats.print_summary();
  stats.export_csv("artifacts/perf/feature_extraction_latency.csv");
}

// ===== Test 2: Normalization Latency =====
void benchmark_normalization(const BenchConfig& cfg, const NormStats& stats_norm) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 2: Normalization Latency\n";
  std::cout << std::string(60, '=') << "\n";
  
  PerfStats stats("normalization");
  
  // Prepare test data
  std::vector<float> test_vec(cfg.L * cfg.C);
  for (int i = 0; i < cfg.L * cfg.C; ++i) {
    test_vec[i] = 10.0f + 0.5f * i;
  }
  
  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        volatile float normalized = (test_vec[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
  }
  
  // Benchmark
  for (int i = 0; i < cfg.iterations; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        test_vec[idx] = (test_vec[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.record_latency(latency_ns);
  }
  
  stats.print_summary();
}

// ===== Test 3: ONNX Inference Latency =====
void benchmark_onnx_inference(const BenchConfig& cfg) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 3: ONNX Inference Latency\n";
  std::cout << std::string(60, '=') << "\n";
  
  LiTCVGSession session(cfg.model_path, cfg.threads);
  PerfStats stats("onnx_inference");
  
  // Prepare input
  std::vector<float> input(cfg.T * cfg.L * cfg.C, 0.5f);
  std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
  
  std::vector<float> out10, out50, out100;
  
  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    session.infer(input.data(), shape, out10, out50, out100);
  }
  
  // Benchmark
  for (int i = 0; i < cfg.iterations; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    session.infer(input.data(), shape, out10, out50, out100);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.record_latency(latency_ns);
  }
  
  stats.print_summary();
  stats.export_csv("artifacts/perf/onnx_inference_latency.csv");
}

// ===== Test 4: End-to-End Pipeline Latency =====
void benchmark_e2e_pipeline(const BenchConfig& cfg, const NormStats& stats_norm) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 4: End-to-End Pipeline Latency\n";
  std::cout << std::string(60, '=') << "\n";
  
  LiTCVGSession session(cfg.model_path, cfg.threads);
  WindowAssembler wa(cfg.T, cfg.L, cfg.C);
  PerfStats stats("e2e_pipeline");
  
  L2Frame prev = make_test_frame(0);
  float cum_vol = 0.0f;
  
  // Prime the window
  for (int i = 0; i < cfg.T; ++i) {
    L2Frame cur = make_test_frame(i);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    
    // Normalize
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat[idx] = (feat[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    wa.push_frame(feat);
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    L2Frame cur = make_test_frame(cfg.T + i);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat[idx] = (feat[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    wa.push_frame(feat);
    
    const float* ptr = wa.view_input_time_ordered();
    std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
    std::vector<float> out10, out50, out100;
    session.infer(ptr, shape, out10, out50, out100);
    
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  // Benchmark
  for (int i = 0; i < cfg.iterations; ++i) {
    L2Frame cur = make_test_frame(cfg.T + cfg.warmup + i);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // 1. Feature extraction
    auto feat = make_feature_vector(prev, cur, cum_vol);
    
    // 2. Normalization
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat[idx] = (feat[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    // 3. Window update
    wa.push_frame(feat);
    
    // 4. Inference
    const float* ptr = wa.view_input_time_ordered();
    std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
    std::vector<float> out10, out50, out100;
    session.infer(ptr, shape, out10, out50, out100);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    stats.record_latency(latency_ns);
    
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  stats.print_summary();
  stats.export_csv("artifacts/perf/e2e_pipeline_latency.csv");
}

// ===== Test 5: Throughput Test =====
void benchmark_throughput(const BenchConfig& cfg, const NormStats& stats_norm) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 5: Throughput Test\n";
  std::cout << std::string(60, '=') << "\n";
  
  LiTCVGSession session(cfg.model_path, cfg.threads);
  WindowAssembler wa(cfg.T, cfg.L, cfg.C);
  
  L2Frame prev = make_test_frame(0);
  float cum_vol = 0.0f;
  
  // Prime the window
  for (int i = 0; i < cfg.T; ++i) {
    L2Frame cur = make_test_frame(i);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat[idx] = (feat[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    wa.push_frame(feat);
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  const int TEST_ITERATIONS = 10000;
  int inferences = 0;
  
  auto t0 = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < TEST_ITERATIONS; ++i) {
    L2Frame cur = make_test_frame(cfg.T + i);
    auto feat = make_feature_vector(prev, cur, cum_vol);
    
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat[idx] = (feat[idx] - stats_norm.mean[c]) / stats_norm.std[c];
      }
    }
    
    wa.push_frame(feat);
    
    const float* ptr = wa.view_input_time_ordered();
    std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
    std::vector<float> out10, out50, out100;
    
    if (session.infer(ptr, shape, out10, out50, out100)) {
      inferences++;
    }
    
    cum_vol += cur.trade_vol;
    prev = cur;
  }
  
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  
  std::cout << "\n📊 Throughput Results:\n";
  std::cout << "   Iterations: " << TEST_ITERATIONS << "\n";
  std::cout << "   Inferences: " << inferences << "\n";
  std::cout << "   Total time: " << elapsed_ms << " ms\n";
  std::cout << "   Throughput: " << (inferences / (elapsed_ms / 1000.0)) << " inferences/sec\n";
  std::cout << "   Avg latency: " << (elapsed_ms / inferences) << " ms/inference\n";
}

// ===== Test 6: Thread Scaling Analysis =====
void benchmark_thread_scaling(const BenchConfig& base_cfg) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST 6: Thread Scaling Analysis\n";
  std::cout << std::string(60, '=') << "\n";
  
  std::vector<int> thread_counts = {1, 2, 4};
  std::vector<double> throughputs;
  
  for (int threads : thread_counts) {
    BenchConfig cfg = base_cfg;
    cfg.threads = threads;
    
    LiTCVGSession session(cfg.model_path, threads);
    
    // Prepare input
    std::vector<float> input(cfg.T * cfg.L * cfg.C, 0.5f);
    std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
    std::vector<float> out10, out50, out100;
    
    // Warmup
    for (int i = 0; i < 50; ++i) {
      session.infer(input.data(), shape, out10, out50, out100);
    }
    
    // Benchmark
    const int iters = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iters; ++i) {
      session.infer(input.data(), shape, out10, out50, out100);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double throughput = iters / elapsed_sec;
    throughputs.push_back(throughput);
    
    std::cout << "   Threads=" << threads << ": " 
              << std::fixed << std::setprecision(2) 
              << throughput << " inferences/sec\n";
  }
  
  // Compute speedup
  std::cout << "\n📈 Speedup vs 1 thread:\n";
  for (size_t i = 0; i < thread_counts.size(); ++i) {
    double speedup = throughputs[i] / throughputs[0];
    std::cout << "   Threads=" << thread_counts[i] << ": " 
              << std::fixed << std::setprecision(2) 
              << speedup << "x\n";
  }
}

// ===== Main =====
int main(int argc, char** argv) {
  BenchConfig cfg;
  
  if (argc >= 2) cfg.model_path = argv[1];
  if (argc >= 3) cfg.norm_path = argv[2];
  
  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════════════╗\n";
  std::cout << "║          LiT-CVG Performance Benchmark Suite         ║\n";
  std::cout << "╚═══════════════════════════════════════════════════════╝\n";
  std::cout << "\nConfiguration:\n";
  std::cout << "  Model: " << cfg.model_path << "\n";
  std::cout << "  Norm:  " << cfg.norm_path << "\n";
  std::cout << "  Shape: (1, " << cfg.T << ", " << cfg.L << ", " << cfg.C << ")\n";
  std::cout << "  Warmup: " << cfg.warmup << " iterations\n";
  std::cout << "  Test iterations: " << cfg.iterations << "\n";
  std::cout << "\n";
  
  // Create output directory
  system("mkdir -p artifacts/perf");
  
  // Load norm stats
  NormStats stats;
  if (!load_norm_stats_json(cfg.norm_path, stats)) {
    std::cerr << "Warning: Using default norm stats\n";
    stats.mean.resize(cfg.C, 0.0f);
    stats.std.resize(cfg.C, 1.0f);
  }
  
  // Run tests
  benchmark_feature_extraction(cfg);
  benchmark_normalization(cfg, stats);
  benchmark_onnx_inference(cfg);
  benchmark_e2e_pipeline(cfg, stats);
  benchmark_throughput(cfg, stats);
  benchmark_thread_scaling(cfg);
  
  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════════════╗\n";
  std::cout << "║              Benchmark Complete! ✅                   ║\n";
  std::cout << "║  Results exported to artifacts/perf/*.csv             ║\n";
  std::cout << "╚═══════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
  
  return 0;
}
