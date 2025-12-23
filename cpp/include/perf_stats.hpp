#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

class PerfStats {
public:
  explicit PerfStats(const std::string& name) : name_(name) {
    latencies_.reserve(100000);  // Pre-allocate for efficiency
  }
  
  // Record a latency sample (in nanoseconds)
  void record_latency(int64_t latency_ns) {
    latencies_.push_back(latency_ns);
  }
  
  // Compute percentiles
  struct Percentiles {
    double p50 = 0.0;   // Median
    double p95 = 0.0;
    double p99 = 0.0;
    double p999 = 0.0;
    double max = 0.0;
    double min = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    size_t count = 0;
  };
  
  Percentiles compute() const {
    if (latencies_.empty()) {
      return Percentiles{};
    }
    
    Percentiles p;
    p.count = latencies_.size();
    
    // Sort for percentile calculation
    std::vector<int64_t> sorted = latencies_;
    std::sort(sorted.begin(), sorted.end());
    
    // Percentiles (in microseconds for readability)
    auto to_us = [](int64_t ns) { return ns / 1000.0; };
    
    p.min = to_us(sorted[0]);
    p.max = to_us(sorted[sorted.size() - 1]);
    p.p50 = to_us(sorted[sorted.size() * 50 / 100]);
    p.p95 = to_us(sorted[sorted.size() * 95 / 100]);
    p.p99 = to_us(sorted[sorted.size() * 99 / 100]);
    p.p999 = to_us(sorted[sorted.size() * 999 / 1000]);
    
    // Mean
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    p.mean = to_us(sum / sorted.size());
    
    // Standard deviation
    double sq_sum = 0.0;
    for (auto lat : sorted) {
      double diff = to_us(lat) - p.mean;
      sq_sum += diff * diff;
    }
    p.stddev = std::sqrt(sq_sum / sorted.size());
    
    return p;
  }
  
  // Print summary
  void print_summary() const {
    auto p = compute();
    
    std::cout << "\n┌─────────────────────────────────────────┐\n";
    std::cout << "│ Performance: " << std::setw(24) << std::left << name_ << " │\n";
    std::cout << "├─────────────────────────────────────────┤\n";
    std::cout << "│ Samples:  " << std::setw(29) << p.count << " │\n";
    std::cout << "│ Mean:     " << std::setw(20) << std::fixed << std::setprecision(2) << p.mean << " µs     │\n";
    std::cout << "│ StdDev:   " << std::setw(20) << p.stddev << " µs     │\n";
    std::cout << "│ Min:      " << std::setw(20) << p.min << " µs     │\n";
    std::cout << "│ P50:      " << std::setw(20) << p.p50 << " µs     │\n";
    std::cout << "│ P95:      " << std::setw(20) << p.p95 << " µs     │\n";
    std::cout << "│ P99:      " << std::setw(20) << p.p99 << " µs     │\n";
    std::cout << "│ P99.9:    " << std::setw(20) << p.p999 << " µs     │\n";
    std::cout << "│ Max:      " << std::setw(20) << p.max << " µs     │\n";
    std::cout << "└─────────────────────────────────────────┘\n";
  }
  
  // Export to CSV
  void export_csv(const std::string& filepath) const {
    std::ofstream f(filepath);
    if (!f.is_open()) {
      std::cerr << "Failed to open " << filepath << "\n";
      return;
    }
    
    f << "latency_us\n";
    for (auto lat_ns : latencies_) {
      f << (lat_ns / 1000.0) << "\n";
    }
    
    std::cout << "📊 Exported " << latencies_.size() << " samples to " << filepath << "\n";
  }
  
  // Get raw data
  const std::vector<int64_t>& raw_data() const { return latencies_; }
  
private:
  std::string name_;
  std::vector<int64_t> latencies_;
};


// ===== System-wide performance monitor =====
class SystemMonitor {
public:
  void start() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  
  void stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
  }
  
  double elapsed_ms() const {
    auto duration = end_time_ - start_time_;
    return std::chrono::duration<double, std::milli>(duration).count();
  }
  
  void print_throughput(int64_t num_operations, const std::string& unit = "ops") const {
    double elapsed_sec = elapsed_ms() / 1000.0;
    if (elapsed_sec > 0) {
      double throughput = num_operations / elapsed_sec;
      std::cout << "📈 Throughput: " << std::fixed << std::setprecision(2) 
                << throughput << " " << unit << "/sec\n";
    }
  }
  
private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
};
