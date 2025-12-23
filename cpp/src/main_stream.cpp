#include "preproc.hpp"
#include "fe_ofi.hpp"
#include "litcvg_session.hpp"
#include "ring_buffer.hpp"
#include "perf_stats.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <string>

// ===== Configuration =====
struct ThreadedConfig {
  std::string model_path;
  std::string norm_path;
  int T = 100;
  int L = 20;
  int C = 7;
  int threads = 1;
  int num_frames = 10000;  // Total frames to process
  bool enable_profiling = true;
};

// ===== Thread-safe LOB frame with metadata =====
struct LOBFrame {
  L2Frame data;
  int64_t timestamp_ns;  // Arrival timestamp
  int seq_num;           // Sequence number for tracking
};

// ===== Feature frame for inference =====
struct FeatureFrame {
  std::vector<float> features;  // (L*C) flattened
  int64_t timestamp_ns;         // Feature extraction completion time
  int seq_num;
};

// ===== Shared state between threads =====
struct SharedState {
  // Ring buffers (sized to prevent blocking)
  RingBuffer<LOBFrame, 4096> lob_queue;      // Market data -> Feature extraction
  RingBuffer<FeatureFrame, 2048> feat_queue; // Feature extraction -> Inference
  
  // Control flags
  std::atomic<bool> running{true};
  std::atomic<bool> market_done{false};
  std::atomic<bool> feature_done{false};
  
  // Performance counters
  std::atomic<int64_t> frames_received{0};
  std::atomic<int64_t> features_extracted{0};
  std::atomic<int64_t> inferences_completed{0};
};

// ===== YAML config loader (simplified) =====
static bool load_threaded_yaml(const std::string& path, ThreadedConfig& cfg) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  
  std::string line;
  while (std::getline(f, line)) {
    auto pos = line.find(':');
    if (pos == std::string::npos) continue;
    
    std::string key = line.substr(0, pos);
    std::string val = line.substr(pos + 1);
    
    // Trim whitespace and quotes
    auto trim = [](std::string& s) {
      while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
      while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
      if (!s.empty() && s.front()=='"') s.erase(s.begin());
      if (!s.empty() && s.back()=='"') s.pop_back();
    };
    trim(key); trim(val);
    
    if (key == "model_path") cfg.model_path = val;
    else if (key == "norm_stats") cfg.norm_path = val;
    else if (key == "T") cfg.T = std::stoi(val);
    else if (key == "L") cfg.L = std::stoi(val);
    else if (key == "C_raw") cfg.C = std::stoi(val);
    else if (key == "threads") cfg.threads = std::stoi(val);
    else if (key == "num_frames") cfg.num_frames = std::stoi(val);
  }
  
  return !cfg.model_path.empty();
}

// ===== Synthetic LOB generator for testing =====
static L2Frame make_synth_frame(float mid, float spread, int seed_offset) {
  L2Frame f;
  float tick = 0.01f;
  float bid0 = mid - spread / 2.0f;
  float ask0 = mid + spread / 2.0f;
  
  for (int i = 0; i < 20; ++i) {
    f.bid_p[i] = bid0 - i * tick;
    f.ask_p[i] = ask0 + i * tick;
    // Add some variation
    f.bid_s[i] = 10.0f + 0.1f * i + 0.5f * (seed_offset % 10);
    f.ask_s[i] = 10.0f + 0.1f * i + 0.5f * ((seed_offset + 5) % 10);
  }
  
  f.trade_vol = 0.5f + 0.1f * (seed_offset % 5);
  return f;
}

// ===== Thread 1: Market Data Receiver =====
void market_data_thread(SharedState& state, const ThreadedConfig& cfg) {
  std::cout << "[MarketData] Thread started (tid=" << std::this_thread::get_id() << ")\n";
  
  int seq = 0;
  float mid = 100.0f;
  float spread = 0.01f;
  
  for (int i = 0; i < cfg.num_frames; ++i) {
    // Simulate data arrival
    mid += 0.0001f * (i % 100 - 50);  // Random walk
    spread = 0.01f + 0.0001f * (i % 20);
    
    LOBFrame lob_frame;
    lob_frame.data = make_synth_frame(mid, spread, i);
    lob_frame.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    lob_frame.seq_num = seq++;
    
    // Push to queue (blocking if full)
    while (!state.lob_queue.push(lob_frame)) {
      std::this_thread::yield();
    }
    
    state.frames_received.fetch_add(1, std::memory_order_relaxed);
    
    // Simulate realistic inter-arrival time (e.g., 1ms)
    // std::this_thread::sleep_for(std::chrono::microseconds(1000));
  }
  
  state.market_done.store(true, std::memory_order_release);
  std::cout << "[MarketData] Thread finished. Frames sent: " << seq << "\n";
}

// ===== Thread 2: Feature Extraction =====
void feature_extraction_thread(SharedState& state, const ThreadedConfig& cfg, 
                               const NormStats& stats) {
  std::cout << "[FeatureExt] Thread started (tid=" << std::this_thread::get_id() << ")\n";
  
  L2Frame prev_frame = make_synth_frame(100.0f, 0.01f, 0);
  float cum_vol = 0.0f;
  int processed = 0;
  
  PerfStats perf("feature_extraction");
  
  while (state.running.load(std::memory_order_acquire)) {
    LOBFrame lob_frame;
    
    // Try to pop from queue
    if (!state.lob_queue.pop(lob_frame)) {
      // Queue empty
      if (state.market_done.load(std::memory_order_acquire)) {
        break;  // Market data finished
      }
      std::this_thread::yield();
      continue;
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Compute features
    auto feat_vec = make_feature_vector(prev_frame, lob_frame.data, cum_vol);
    cum_vol += lob_frame.data.trade_vol;
    
    // Normalize (in-place modification of feat_vec)
    for (int l = 0; l < cfg.L; ++l) {
      for (int c = 0; c < cfg.C; ++c) {
        int idx = l * cfg.C + c;
        feat_vec[idx] = (feat_vec[idx] - stats.mean[c]) / stats.std[c];
      }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    perf.record_latency(latency_ns);
    
    // Package for inference
    FeatureFrame ff;
    ff.features = std::move(feat_vec);
    ff.timestamp_ns = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    ff.seq_num = lob_frame.seq_num;
    
    // Push to inference queue
    while (!state.feat_queue.push(ff)) {
      std::this_thread::yield();
    }
    
    state.features_extracted.fetch_add(1, std::memory_order_relaxed);
    prev_frame = lob_frame.data;
    processed++;
  }
  
  state.feature_done.store(true, std::memory_order_release);
  std::cout << "[FeatureExt] Thread finished. Features extracted: " << processed << "\n";
  perf.print_summary();
}

// ===== Thread 3: Model Inference =====
void inference_thread(SharedState& state, const ThreadedConfig& cfg, 
                     LiTCVGSession& session) {
  std::cout << "[Inference] Thread started (tid=" << std::this_thread::get_id() << ")\n";
  
  WindowAssembler wa(cfg.T, cfg.L, cfg.C);
  int inferences = 0;
  
  PerfStats perf("inference");
  PerfStats e2e_perf("end_to_end");
  
  while (state.running.load(std::memory_order_acquire)) {
    FeatureFrame ff;
    
    // Try to pop from queue
    if (!state.feat_queue.pop(ff)) {
      // Queue empty
      if (state.feature_done.load(std::memory_order_acquire)) {
        break;  // Feature extraction finished
      }
      std::this_thread::yield();
      continue;
    }
    
    // Push to window
    wa.push_frame(ff.features);
    
    const float* input_ptr = wa.view_input_time_ordered();
    if (input_ptr == nullptr) {
      continue;  // Window not full yet
    }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // Run inference
    std::array<int64_t, 4> shape{1, cfg.T, cfg.L, cfg.C};
    std::vector<float> out10, out50, out100;
    
    bool success = session.infer(input_ptr, shape, out10, out50, out100);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    
    if (success) {
      perf.record_latency(latency_ns);
      
      // Compute end-to-end latency (from feature arrival)
      int64_t e2e_ns = t1.time_since_epoch().count() - ff.timestamp_ns;
      e2e_perf.record_latency(e2e_ns);
      
      inferences++;
      state.inferences_completed.fetch_add(1, std::memory_order_relaxed);
      
      // Optional: extract signal from logits
      // int signal = extract_signal(out10, out50, out100);
    }
  }
  
  std::cout << "[Inference] Thread finished. Inferences: " << inferences << "\n";
  perf.print_summary();
  e2e_perf.print_summary();
}

// ===== Main =====
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " path/to/config.yaml\n";
    return 1;
  }
  
  // Load config
  ThreadedConfig cfg;
  if (!load_threaded_yaml(argv[1], cfg)) {
    std::cerr << "Failed to load config: " << argv[1] << "\n";
    return 1;
  }
  
  std::cout << "=== Threaded Pipeline Configuration ===\n";
  std::cout << "Model: " << cfg.model_path << "\n";
  std::cout << "Norm:  " << cfg.norm_path << "\n";
  std::cout << "Shape: (1, " << cfg.T << ", " << cfg.L << ", " << cfg.C << ")\n";
  std::cout << "Frames: " << cfg.num_frames << "\n";
  std::cout << "========================================\n\n";
  
  // Load norm stats
  NormStats stats;
  if (!load_norm_stats_json(cfg.norm_path, stats)) {
    std::cerr << "Warning: Failed to load norm stats\n";
    stats.mean.resize(cfg.C, 0.0f);
    stats.std.resize(cfg.C, 1.0f);
  }
  
  // Initialize model session
  LiTCVGSession session(cfg.model_path, cfg.threads);
  
  // Shared state
  SharedState state;
  
  std::cout << "🚀 Starting threaded pipeline...\n\n";
  
  auto pipeline_start = std::chrono::high_resolution_clock::now();
  
  // Launch threads
  std::thread t1(market_data_thread, std::ref(state), std::cref(cfg));
  std::thread t2(feature_extraction_thread, std::ref(state), std::cref(cfg), std::cref(stats));
  std::thread t3(inference_thread, std::ref(state), std::cref(cfg), std::ref(session));
  
  // Monitor progress (optional)
  while (state.running.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    int64_t received = state.frames_received.load(std::memory_order_relaxed);
    int64_t extracted = state.features_extracted.load(std::memory_order_relaxed);
    int64_t inferred = state.inferences_completed.load(std::memory_order_relaxed);
    
    std::cout << "[Progress] Received: " << received 
              << " | Extracted: " << extracted 
              << " | Inferred: " << inferred << "\r" << std::flush;
    
    // Check if all threads finished
    if (state.market_done && state.feature_done && 
        state.inferences_completed >= (cfg.num_frames - cfg.T)) {
      state.running.store(false, std::memory_order_release);
      break;
    }
  }
  
  std::cout << "\n\n⏸️  Waiting for threads to finish...\n";
  
  // Join threads
  t1.join();
  t2.join();
  t3.join();
  
  auto pipeline_end = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(pipeline_end - pipeline_start).count();
  
  // Final statistics
  std::cout << "\n=== Pipeline Statistics ===\n";
  std::cout << "Total time: " << total_ms << " ms\n";
  std::cout << "Frames received: " << state.frames_received.load() << "\n";
  std::cout << "Features extracted: " << state.features_extracted.load() << "\n";
  std::cout << "Inferences completed: " << state.inferences_completed.load() << "\n";
  
  if (total_ms > 0) {
    double throughput = state.inferences_completed.load() / (total_ms / 1000.0);
    std::cout << "Throughput: " << throughput << " inferences/sec\n";
  }
  
  std::cout << "===========================\n";
  
  return 0;
}