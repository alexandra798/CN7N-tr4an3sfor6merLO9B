#pragma once
#include <vector>

struct TCAConfig { float fee = 0.0f; float slip_bps = 0.0f; float half_spread = 0.0f; };
struct TCAResult { double pnl = 0.0; double sharpe = 0.0; double turnover = 0.0; };

TCAResult backtest_multi_horizon(const std::vector<int>& signal_10,
                                 const std::vector<int>& signal_50,
                                 const std::vector<int>& signal_100,
                                 const TCAConfig& cfg);
