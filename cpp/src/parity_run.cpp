#include "npy.hpp"
#include "litcvg_session.hpp"
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
  // usage: parity_run <model.onnx> <inputs.npy> <outdir>
  if (argc < 4) {
    std::cerr << "usage: parity_run model.onnx artifacts/golden/inputs.npy artifacts/golden/cpp_out\n";
    return 1;
  }
  std::string model = argv[1];
  std::string in_path = argv[2];
  std::string outdir = argv[3];
  std::filesystem::create_directories(outdir);

  NpyArray arr;
  if (!npy_load_f32(in_path, arr)) {
    std::cerr << "failed to load " << in_path << "\n";
    return 1;
  }
  if (arr.shape.size() != 4) {
    std::cerr << "expected 4D inputs (N,T,L,C)\n";
    return 1;
  }
  const int64_t N = arr.shape[0];
  const int64_t T = arr.shape[1];
  const int64_t L = arr.shape[2];
  const int64_t C = arr.shape[3];
  const size_t per = static_cast<size_t>(T * L * C);

  LiTCVGSession sess(model, 1);

  std::vector<float> out10_all;
  std::vector<float> out50_all;
  std::vector<float> out100_all;
  out10_all.reserve(static_cast<size_t>(N) * 3);
  out50_all.reserve(static_cast<size_t>(N) * 3);
  out100_all.reserve(static_cast<size_t>(N) * 3);

  for (int64_t i = 0; i < N; ++i) {
    const float* x = arr.data.data() + static_cast<size_t>(i) * per;
    std::array<int64_t,4> shape{1, T, L, C};
    std::vector<float> o10, o50, o100;
    if (!sess.infer(x, shape, o10, o50, o100)) {
      std::cerr << "infer failed at i=" << i << "\n";
      return 1;
    }
    if (o10.size() != 3 || o50.size() != 3 || o100.size() != 3) {
      std::cerr << "unexpected logits size\n";
      return 1;
    }
    out10_all.insert(out10_all.end(), o10.begin(), o10.end());
    out50_all.insert(out50_all.end(), o50.begin(), o50.end());
    out100_all.insert(out100_all.end(), o100.begin(), o100.end());
  }

  std::vector<int64_t> shp{N, 3};
  if (!npy_save_f32(outdir + "/logits_10.npy", shp, out10_all.data(), out10_all.size())) return 1;
  if (!npy_save_f32(outdir + "/logits_50.npy", shp, out50_all.data(), out50_all.size())) return 1;
  if (!npy_save_f32(outdir + "/logits_100.npy", shp, out100_all.data(), out100_all.size())) return 1;

  std::cout << "saved cpp logits to " << outdir << "\n";
  return 0;
}
