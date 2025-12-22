#include "litcvg_session.hpp"

LiTCVGSession::LiTCVGSession(const std::string& onnx, int threads)
    : ort_(onnx, threads) {
  out_names_ = {"logits_10", "logits_50", "logits_100"};
}

bool LiTCVGSession::infer(const float* input_1_T_L_C,
                          const std::array<int64_t, 4>& shape,
                          std::vector<float>& out10,
                          std::vector<float>& out50,
                          std::vector<float>& out100) {
  auto outs = ort_.infer(input_1_T_L_C, shape, out_names_);
  if (outs.size() != 3) return false;
  out10 = std::move(outs[0]);
  out50 = std::move(outs[1]);
  out100 = std::move(outs[2]);
  return true;
}
