#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct NpyArray {
  std::vector<int64_t> shape;
  std::vector<float> data; // float32 only
};

// Reads little-endian float32 .npy (v1.0/v2.0) C-order, not Fortran order.
bool npy_load_f32(const std::string& path, NpyArray& out);

// Writes little-endian float32 .npy v1.0, C-order
bool npy_save_f32(const std::string& path, const std::vector<int64_t>& shape, const float* data, size_t n);
