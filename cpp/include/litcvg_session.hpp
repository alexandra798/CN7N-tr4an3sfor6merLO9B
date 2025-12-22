#pragma once
#include "session.hpp"
#include <string>
#include <vector>

class LiTCVGSession {
public:
  explicit LiTCVGSession(const std::string& onnx, int threads = 1);

  bool infer(const float* input_1_T_L_C,
             const std::array<int64_t, 4>& shape,
             std::vector<float>& out10,
             std::vector<float>& out50,
             std::vector<float>& out100);

private:
  OrtSessionWrapper ort_;
  std::vector<const char*> out_names_;
};
