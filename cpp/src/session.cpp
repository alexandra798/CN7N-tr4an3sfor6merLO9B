#include "session.hpp"
#include <stdexcept>

OrtSessionWrapper::OrtSessionWrapper(const std::string& model_path, int intra_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "deeplob_hft") {
  opts_.SetIntraOpNumThreads(intra_threads);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = Ort::Session(env_, model_path.c_str(), opts_);

  // cache input name(s)
  size_t n_in = session_.GetInputCount();
  input_names_.reserve(n_in);
  for (size_t i = 0; i < n_in; ++i) {

    auto name_alloc = session_.GetInputNameAllocated(i, alloc_);
    input_names_.push_back(name_alloc.get());
    input_name_holders_.push_back(std::move(name_alloc));



  }
}

std::vector<std::vector<float>> OrtSessionWrapper::infer(
    const float* input,
    const std::array<int64_t, 4>& shape,
    const std::vector<const char*>& out_names) {

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  size_t n_elem = 1;
  for (auto d : shape) n_elem *= static_cast<size_t>(d);

  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem, const_cast<float*>(input), n_elem, shape.data(), shape.size());

  // run
  auto outputs = session_.Run(Ort::RunOptions{nullptr},
                             input_names_.data(), &in_tensor, 1,
                             out_names.data(), out_names.size());

  std::vector<std::vector<float>> out_flat;
  out_flat.reserve(outputs.size());
  for (auto& o : outputs) {
    float* p = o.GetTensorMutableData<float>();
    auto info = o.GetTensorTypeAndShapeInfo();
    size_t sz = info.GetElementCount();
    out_flat.emplace_back(p, p + sz);
  }
  return out_flat;
}
