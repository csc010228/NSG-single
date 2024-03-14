#pragma once

#include <algorithm>
#include <random>
#include <unordered_set>

namespace glass {

// 随机生成一个长度为size，元素的取值范围是[0, N - 1]的，按照严格升序排序的int类型数组
inline void GenRandom(std::mt19937 &rng, int *addr, const int size,
                      const int N) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % N;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() {
    return int64_t(rand_int()) | int64_t(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return rand_int() % max; }

  /// between 0 and 1
  // float rand_float() { return mt() / float(mt.max()); }

  // double rand_double() { return mt() / double(mt.max()); }
};

} // namespace glass