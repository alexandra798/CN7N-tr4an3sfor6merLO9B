#include "tca.hpp"
#include <cmath>

TCAResult backtest_multi_horizon(const std::vector<int>& s10,
                                 const std::vector<int>& s50,
                                 const std::vector<int>& s100,
                                 const TCAConfig& cfg) {
  // Placeholder: combine signals and compute toy pnl.
  // Replace with realistic matching, fees, slippage, and latency sensitivity.
  const size_t n = std::min({s10.size(), s50.size(), s100.size()});
  double pnl = 0.0;
  double turnover = 0.0;
  int pos = 0;
  for (size_t i = 0; i < n; ++i) {
    int sig = 0;
    sig += (s10[i] == 2) - (s10[i] == 0);
    sig += (s50[i] == 2) - (s50[i] == 0);
    sig += (s100[i] == 2) - (s100[i] == 0);
    int new_pos = (sig > 0) ? 1 : (sig < 0) ? -1 : 0;
    turnover += std::abs(new_pos - pos);
    // toy pnl: reward holding correct direction; penalize turnover
    pnl += 0.1 * new_pos - cfg.fee * std::abs(new_pos - pos);
    pos = new_pos;
  }
  TCAResult r;
  r.pnl = pnl;
  r.turnover = turnover;
  r.sharpe = (turnover > 0.0) ? pnl / std::sqrt(turnover) : 0.0;
  return r;
}
