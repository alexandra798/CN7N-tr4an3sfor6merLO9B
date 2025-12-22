#pragma once
#include "/Users/avelynekim/onnxruntime/include/onnxruntime_cxx_api.h"
#include <array>
#include <vector>
#include <string>
#include <cstdint>

class OrtSessionWrapper {
public:
  explicit OrtSessionWrapper(const std::string& model_path, int intra_threads = 1);

  std::vector<std::vector<float>> infer(const float* input,
                                        const std::array<int64_t, 4>& shape,
                                        const std::vector<const char*>& out_names);

  const std::vector<const char*>& input_names() const { return input_names_; }

private:
  Ort::Env env_;
  Ort::SessionOptions opts_;
  Ort::Session session_{nullptr};
  Ort::AllocatorWithDefaultOptions alloc_;
  std::vector<const char*> input_names_;
  std::vector<Ort::AllocatedStringPtr> input_name_holders_;
};
