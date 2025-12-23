#pragma once
#include <vector>
#include <string>

// ===== Configuration =====

struct TCAConfig { 
  float fee = 0.0f;             // Transaction fee per unit
  float slip_bps = 0.0f;        // Base slippage in basis points
  float half_spread = 0.0f;     // Half of bid-ask spread
  int latency_us = 0;           // Execution latency in microseconds
};

// ===== Execution Result =====

struct ExecutionResult {
  double fill_price = 0.0;      // Actual execution price
  double slippage = 0.0;        // Slippage in basis points
  double market_impact = 0.0;   // Market impact in basis points
  int64_t latency_ns = 0;       // Execution latency in nanoseconds
};

// ===== TCA Result =====

struct TCAResult { 
  double pnl = 0.0;             // Net profit & loss
  double sharpe = 0.0;          // Sharpe ratio (annualized)
  double turnover = 0.0;        // Total turnover (sum of |position changes|)
};

// ===== Main TCA Function =====

/**
 * Backtest multi-horizon trading strategy with realistic TCA.
 * 
 * @param signal_10  Signal predictions for 10-step horizon (0=down, 1=flat, 2=up)
 * @param signal_50  Signal predictions for 50-step horizon
 * @param signal_100 Signal predictions for 100-step horizon
 * @param cfg        TCA configuration (fees, slippage, latency)
 * @return TCAResult containing PnL, Sharpe, and turnover metrics
 * 
 * Features:
 * - Weighted signal combination (short-term priority)
 * - Realistic slippage model (proportional to spread + order size)
 * - Market impact modeling (square-root of order flow)
 * - Latency simulation
 * - Transaction cost accounting
 */
TCAResult backtest_multi_horizon(const std::vector<int>& signal_10,
                                 const std::vector<int>& signal_50,
                                 const std::vector<int>& signal_100,
                                 const TCAConfig& cfg);

// ===== Latency Sensitivity Analysis =====

/**
 * Run TCA backtest with varying latency values.
 * 
 * Useful for understanding how execution latency affects profitability.
 * 
 * @param signal_10, signal_50, signal_100  Signal predictions
 * @param base_cfg                          Base configuration
 * @param latency_us_values                 Vector of latency values to test (microseconds)
 * @return Vector of TCAResult, one per latency value
 * 
 * Example usage:
 *   std::vector<int> latencies = {0, 50, 100, 200, 500, 1000};  // µs
 *   auto results = latency_sensitivity_sweep(s10, s50, s100, cfg, latencies);
 */
std::vector<TCAResult> latency_sensitivity_sweep(
    const std::vector<int>& signal_10,
    const std::vector<int>& signal_50,
    const std::vector<int>& signal_100,
    const TCAConfig& base_cfg,
    const std::vector<int>& latency_us_values);

// ===== Export Utilities =====

/**
 * Export TCA results to CSV for analysis.
 * 
 * CSV format:
 *   latency_us, pnl, sharpe, turnover
 */
void export_tca_results_csv(const std::string& path,
                            const std::vector<TCAResult>& results,
                            const std::vector<int>& latency_values);