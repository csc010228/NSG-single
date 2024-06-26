#pragma once

#include <string>
#include <unordered_map>

namespace nsg {

enum class Metric {
  L2,
  IP,
};

inline constexpr size_t upper_div(size_t x, size_t y) {
  return (x + y - 1) / y;
}

inline constexpr int64_t do_align(int64_t x, int64_t align) {
  return (x + align - 1) / align * align;
}

#if defined(__clang__)

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define FAST_BEGIN                                                             \
  _Pragma("GCC push_options") _Pragma(                                         \
      "GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAST_END _Pragma("GCC pop_options")
#define GLASS_INLINE [[gnu::always_inline]]
#else

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE

#endif

} // namespace nsg