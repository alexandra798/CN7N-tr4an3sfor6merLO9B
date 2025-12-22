#pragma once
#include <atomic>
#include <cstddef>

template <typename T, size_t N>
class RingBuffer {
public:
  bool push(const T& v) {
    size_t w = w_.load(std::memory_order_relaxed);
    size_t r = r_.load(std::memory_order_acquire);
    if ((w + 1) % N == r) return false; // full
    data_[w] = v;
    w_.store((w + 1) % N, std::memory_order_release);
    return true;
  }

  bool pop(T& v) {
    size_t r = r_.load(std::memory_order_relaxed);
    size_t w = w_.load(std::memory_order_acquire);
    if (r == w) return false; // empty
    v = data_[r];
    r_.store((r + 1) % N, std::memory_order_release);
    return true;
  }

private:
  alignas(64) T data_[N];
  std::atomic<size_t> w_{0}, r_{0};
};
