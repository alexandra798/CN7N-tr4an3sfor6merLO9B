#include "tca.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// ===== Helper functions =====

static double compute_mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double compute_std(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  double mean = compute_mean(v);
  double sq_sum = 0.0;
  for (auto x : v) {
    sq_sum += (x - mean) * (x - mean);
  }
  return std::sqrt(sq_sum / (v.size() - 1));
}

// ===== Order Matching Simulator =====

struct LOBState {
  double bid_p;     // Best bid price
  double ask_p;     // Best ask price
  double bid_s;     // Best bid size
  double ask_s;     // Best ask size
  double mid;       // Mid price
  double spread;    // Bid-ask spread
};

class OrderMatcher {
public:
  explicit OrderMatcher(const TCAConfig& cfg) : cfg_(cfg) {}
  
  // Execute a market order with realistic slippage and impact
  ExecutionResult match_order(const LOBState& lob, int side, double size) {
    ExecutionResult result;
    result.latency_ns = static_cast<int64_t>(cfg_.latency_us) * 1000;
    
    // Determine execution price based on side
    double base_price = (side > 0) ? lob.ask_p : lob.bid_p;
    
    // Slippage model: proportional to spread and size
    // Larger orders suffer more slippage
    double slippage_bps = cfg_.slip_bps;
    
    // Market impact: square-root model
    // impact ∝ sqrt(order_size / available_liquidity)
    double available_liquidity = (side > 0) ? lob.ask_s : lob.bid_s;
    double impact_ratio = std::sqrt(size / (available_liquidity + 1e-8));
    double market_impact_bps = cfg_.slip_bps * impact_ratio * 0.5;  // 50% of slippage
    
    // Total slippage in bps
    double total_slippage_bps = slippage_bps + market_impact_bps;
    result.slippage = total_slippage_bps;
    result.market_impact = market_impact_bps;
    
    // Apply slippage to execution price
    // For buy: price increases; for sell: price decreases
    double slippage_factor = 1.0 + (side > 0 ? 1.0 : -1.0) * total_slippage_bps / 10000.0;
    result.fill_price = base_price * slippage_factor;
    
    return result;
  }

private:
  const TCAConfig& cfg_;
};

// ===== Signal Combination Strategies =====

int combine_signals_voting(int s10, int s50, int s100) {
  // Simple voting: majority wins
  int vote_down = (s10 == 0) + (s50 == 0) + (s100 == 0);
  int vote_flat = (s10 == 1) + (s50 == 1) + (s100 == 1);
  int vote_up = (s10 == 2) + (s50 == 2) + (s100 == 2);
  
  if (vote_up > vote_down && vote_up > vote_flat) return 2;
  if (vote_down > vote_up && vote_down > vote_flat) return 0;
  return 1;  // flat or tie
}

int combine_signals_weighted(int s10, int s50, int s100) {
  // Weighted by horizon: shorter horizons have higher weight
  // w10=0.5, w50=0.3, w100=0.2
  double score = 0.0;
  score += 0.5 * ((s10 == 2) ? 1.0 : (s10 == 0) ? -1.0 : 0.0);
  score += 0.3 * ((s50 == 2) ? 1.0 : (s50 == 0) ? -1.0 : 0.0);
  score += 0.2 * ((s100 == 2) ? 1.0 : (s100 == 0) ? -1.0 : 0.0);
  
  if (score > 0.3) return 2;   // up
  if (score < -0.3) return 0;  // down
  return 1;  // flat
}

int combine_signals_short_priority(int s10, int s50, int s100) {
  // Priority to shortest horizon (most tradeable)
  if (s10 == 0 || s10 == 2) return s10;
  if (s50 == 0 || s50 == 2) return s50;
  return s100;
}

// ===== Main TCA Backtest =====

TCAResult backtest_multi_horizon(const std::vector<int>& s10,
                                 const std::vector<int>& s50,
                                 const std::vector<int>& s100,
                                 const TCAConfig& cfg) {
  const size_t n = std::min({s10.size(), s50.size(), s100.size()});
  
  if (n == 0) {
    return TCAResult{};
  }
  
  OrderMatcher matcher(cfg);
  
  // Trading state
  int position = 0;
  double cash = 0.0;
  double last_fill_price = 100.0;  // Initial reference price
  
  std::vector<double> returns;
  std::vector<double> pnl_series;
  double total_turnover = 0.0;
  double total_fees = 0.0;
  double total_slippage_cost = 0.0;
  
  // Simulate LOB states (simplified: assume constant spread and mid evolution)
  for (size_t i = 0; i < n; ++i) {
    // Simulate LOB (in real system, read from historical data)
    LOBState lob;
    lob.mid = 100.0 + 0.001 * static_cast<double>(i);  // Slight drift
    lob.spread = cfg.half_spread * 2.0;
    lob.bid_p = lob.mid - cfg.half_spread;
    lob.ask_p = lob.mid + cfg.half_spread;
    lob.bid_s = 100.0;  // Dummy liquidity
    lob.ask_s = 100.0;
    
    // Combine signals
    int signal = combine_signals_weighted(s10[i], s50[i], s100[i]);
    
    // Convert signal to target position
    int target_pos = 0;
    if (signal == 2) target_pos = 1;       // up -> long
    else if (signal == 0) target_pos = -1; // down -> short
    else target_pos = 0;                    // flat -> neutral
    
    // Execute trade if position changes
    if (target_pos != position) {
      int delta = target_pos - position;
      double trade_size = std::abs(delta);
      int side = (delta > 0) ? 1 : -1;  // 1=buy, -1=sell
      
      // Execute order
      ExecutionResult exec = matcher.match_order(lob, side, trade_size);
      
      // Update accounting
      double trade_value = exec.fill_price * trade_size;
      double fee = cfg.fee * trade_size;
      double slippage_cost = (exec.slippage / 10000.0) * exec.fill_price * trade_size;
      
      cash -= side * trade_value;
      cash -= fee;
      total_fees += fee;
      total_slippage_cost += slippage_cost;
      total_turnover += trade_size;
      
      // Update position
      position = target_pos;
      last_fill_price = exec.fill_price;
    }
    
    // Mark-to-market PnL
    double mtm_price = lob.mid;
    double unrealized_pnl = position * (mtm_price - last_fill_price);
    double total_pnl = cash + unrealized_pnl;
    
    pnl_series.push_back(total_pnl);
    
    // Record return
    if (i > 0) {
      double ret = (pnl_series[i] - pnl_series[i-1]) / (std::abs(pnl_series[i-1]) + 1.0);
      returns.push_back(ret);
    }
  }
  
  // Close final position
  if (position != 0) {
    total_turnover += std::abs(position);
  }
  
  // Compute metrics
  TCAResult result;
  result.pnl = pnl_series.empty() ? 0.0 : pnl_series.back();
  result.turnover = total_turnover;
  
  // Sharpe ratio (annualized)
  if (returns.size() > 1) {
    double mean_ret = compute_mean(returns);
    double std_ret = compute_std(returns);
    
    if (std_ret > 1e-8) {
      // Assuming returns are per-step; annualize assuming 252 days, 6.5 hours, 3600 steps/hour
      double steps_per_year = 252.0 * 6.5 * 3600.0;
      result.sharpe = mean_ret / std_ret * std::sqrt(steps_per_year);
    } else {
      result.sharpe = 0.0;
    }
  } else {
    result.sharpe = 0.0;
  }
  
  return result;
}

// ===== Latency Sensitivity Analysis =====

std::vector<TCAResult> latency_sensitivity_sweep(
    const std::vector<int>& s10,
    const std::vector<int>& s50,
    const std::vector<int>& s100,
    const TCAConfig& base_cfg,
    const std::vector<int>& latency_us_values) {
  
  std::vector<TCAResult> results;
  results.reserve(latency_us_values.size());
  
  for (int latency_us : latency_us_values) {
    TCAConfig cfg = base_cfg;
    cfg.latency_us = latency_us;
    
    TCAResult res = backtest_multi_horizon(s10, s50, s100, cfg);
    results.push_back(res);
  }
  
  return results;
}

// ===== Export for Analysis =====

void export_tca_results_csv(const std::string& path,
                            const std::vector<TCAResult>& results,
                            const std::vector<int>& latency_values) {
  // Placeholder: write CSV
  // Format: latency_us, pnl, sharpe, turnover
  // Implement using std::ofstream
}