/*
NSG
======================================================================================================

    NSG算法
    参考: Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph

======================================================================================================
得分：1577.0071
*/


/* =================================================================================================== */
/*                                             common.hpp                                              */
#include <string>
#include <unordered_map>

#define DEBUG_ENABLED // 定义调试输出开关

namespace glass {

enum class Metric {
  L2,
  IP,
};

inline std::unordered_map<std::string, Metric> metric_map;

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

} // namespace glass
/* =================================================================================================== */
/*                                             memory.hpp                                              */
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

namespace glass {

template <typename T> struct align_alloc {
  T *ptr = nullptr;
  using value_type = T;
  T *allocate(int n) {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(T) + 63) >> 6 << 6;
      return ptr = (T *)aligned_alloc(64, sz);
    }
    int sz = (n * sizeof(T) + (1 << 21) - 1) >> 21 << 21;
    ptr = (T *)aligned_alloc(1 << 21, sz);
    return ptr;
  }
  void deallocate(T *, int) { free(ptr); }
  template <typename U> struct rebind {
    typedef align_alloc<U> other;
  };
  bool operator!=(const align_alloc &rhs) { return ptr != rhs.ptr; }
};

inline void *alloc2M(size_t nbytes) {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
//   auto p = std::aligned_alloc(1 << 21, len);
  auto p = aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline void *alloc64B(size_t nbytes) {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
//   auto p = std::aligned_alloc(1 << 6, len);
  auto p = aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}

} // namespace glass
/* =================================================================================================== */
/*                                            neighbor.hpp                                             */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <vector>

namespace glass {

namespace searcher {

template <typename Block = uint64_t> struct Bitset {
  constexpr static int block_size = sizeof(Block) * 8;
  int nbytes;
  Block *data;
  explicit Bitset(int n)
      : nbytes((n + block_size - 1) / block_size * sizeof(Block)),
        data((uint64_t *)alloc64B(nbytes)) {
    memset(data, 0, nbytes);
  }
  ~Bitset() { free(data); }
  void set(int i) {
    data[i / block_size] |= (Block(1) << (i & (block_size - 1)));
  }
  bool get(int i) {
    return (data[i / block_size] >> (i & (block_size - 1))) & 1;
  }

  void *block_address(int i) { return data + i / block_size; }
};

template <typename dist_t = float> struct Neighbor {
  int id;
  dist_t distance;

  Neighbor() = default;
  Neighbor(int id, dist_t distance) : id(id), distance(distance) {}

  inline friend bool operator<(const Neighbor &lhs, const Neighbor &rhs) {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
  }
  inline friend bool operator>(const Neighbor &lhs, const Neighbor &rhs) {
    return !(lhs < rhs);
  }
};

template <typename dist_t> struct MaxHeap {
  explicit MaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  void push(int u, dist_t dist) {
    if (size < capacity) {
      pool[size] = {u, dist};
      std::push_heap(pool.begin(), pool.begin() + ++size);
    } else if (dist < pool[0].distance) {
      sift_down(0, u, dist);
    }
  }
  int pop() {
    std::pop_heap(pool.begin(), pool.begin() + size--);
    return pool[size].id;
  }
  void sift_down(int i, int u, dist_t dist) {
    pool[0] = {u, dist};
    for (; 2 * i + 1 < size;) {
      int j = i;
      int l = 2 * i + 1, r = 2 * i + 2;
      if (pool[l].distance > dist) {
        j = l;
      }
      if (r < size && pool[r].distance > std::max(pool[l].distance, dist)) {
        j = r;
      }
      if (i == j) {
        break;
      }
      pool[i] = pool[j];
      i = j;
    }
    pool[i] = {u, dist};
  }
  int size = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

template <typename dist_t> struct MinMaxHeap {
  explicit MinMaxHeap(int capacity) : capacity(capacity), pool(capacity) {}
  bool push(int u, dist_t dist) {
    if (cur == capacity) {
      if (dist >= pool[0].distance) {
        return false;
      }
      if (pool[0].id >= 0) {
        size--;
      }
      std::pop_heap(pool.begin(), pool.begin() + cur--);
    }
    pool[cur] = {u, dist};
    std::push_heap(pool.begin(), pool.begin() + ++cur);
    size++;
    return true;
  }
  dist_t max() { return pool[0].distance; }
  void clear() { size = cur = 0; }

  int pop_min() {
    int i = cur - 1;
    for (; i >= 0 && pool[i].id == -1; --i)
      ;
    if (i == -1) {
      return -1;
    }
    int imin = i;
    dist_t vmin = pool[i].distance;
    for (; --i >= 0;) {
      if (pool[i].id != -1 && pool[i].distance < vmin) {
        vmin = pool[i].distance;
        imin = i;
      }
    }
    int ret = pool[imin].id;
    pool[imin].id = -1;
    --size;
    return ret;
  }

  int size = 0, cur = 0, capacity;
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> pool;
};

// 以升序存储距离元素的线性池
template <typename dist_t> struct LinearPool {
  LinearPool(int n, int capacity, int = 0)
      : nb(n), capacity_(capacity), data_(capacity_ + 1), vis(n) {}

  // 二分法查找距离 dist 在池子中的索引
  int find_bsearch(dist_t dist) {
    int lo = 0, hi = size_;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (data_[mid].distance > dist) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  // 将编号为 u 的数据插入到该池子中，u 到查询点 q 的距离为 dist
  // 插入后池子中的元素保持按照距离的升序排序
  // 插入后位置指针指向第一个未被访问过的元素
  bool insert(int u, dist_t dist) {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo],
                 (size_ - lo) * sizeof(Neighbor<dist_t>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
    return true;
  }

  int pop() {
    set_checked(data_[cur_].id);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      cur_++;
    }
    return get_id(data_[pre].id);
  }

  bool has_next() const { return cur_ < size_; }
  int id(int i) const { return get_id(data_[i].id); }
  int size() const { return size_; }
  int capacity() const { return capacity_; }

  constexpr static int kMask = 2147483647;
  int get_id(int id) const { return id & kMask; }
  void set_checked(int &id) { id |= 1 << 31; }
  bool is_checked(int id) { return id >> 31 & 1; }

  int nb;                   // 数据集大小
  int size_ = 0;            // 当前的数据量
  int cur_ = 0;             // 当前位置指针
  int capacity_;            // 最大数据量
  std::vector<Neighbor<dist_t>, align_alloc<Neighbor<dist_t>>> data_;
  Bitset<uint64_t> vis;     // 数据是否被访问过
};

template <typename dist_t> struct HeapPool {
  HeapPool(int n, int capacity, int topk)
      : nb(n), capacity_(capacity), candidates(capacity), retset(topk), vis(n) {
  }
  bool insert(int u, dist_t dist) {
    retset.push(u, dist);
    return candidates.push(u, dist);
  }
  int pop() { return candidates.pop_min(); }
  bool has_next() const { return candidates.size > 0; }
  int id(int i) const { return retset.pool[i].id; }
  int capacity() const { return capacity_; }
  int nb, size_ = 0, capacity_;
  MinMaxHeap<dist_t> candidates;
  MaxHeap<dist_t> retset;
  Bitset<uint64_t> vis;
};

} // namespace searcher

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct Node {
  int id;
  float distance;

  Node() = default;
  Node(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const Node &other) const {
    return distance < other.distance;
  }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance > nn.distance) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance < nn.distance) {
      break;
    }
    if (addr[left].id == nn.id) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

} // namespace glass
/* =================================================================================================== */
/*                                             utils.hpp                                               */
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
};

} // namespace glass
/* =================================================================================================== */
/*                                          simd/avx2.hpp                                              */
#if defined(__AVX2__)

#include <cstdint>
#include <immintrin.h>

namespace glass {

inline float reduce_add_f32x8(__m256 x) {
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x8(__m256i x) {
  auto sumh =
      _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  auto tmp2 = _mm_hadd_epi32(sumh, sumh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

inline int32_t reduce_add_i16x16(__m256i x) {
  auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0),
                            _mm256_extracti128_si256(x, 1));
  auto tmp = _mm256_cvtepi16_epi32(sumh);
  auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0),
                             _mm256_extracti128_si256(tmp, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace glass

#endif
/* =================================================================================================== */
/*                                           simd/avx512.hpp                                           */
#if defined(__AVX512F__)

#include <cstdint>
#include <immintrin.h>

namespace glass {

inline float reduce_add_f32x16(__m512 x) {
  auto sumh =
      _mm256_add_ps(_mm512_castps512_ps256(x), _mm512_extractf32x8_ps(x, 1));
  auto sumhh =
      _mm_add_ps(_mm256_castps256_ps128(sumh), _mm256_extractf128_ps(sumh, 1));
  auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

inline int32_t reduce_add_i32x16(__m512i x) {
  auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(x, 0),
                               _mm512_extracti32x8_epi32(x, 1));
  auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh),
                             _mm256_extracti128_si256(sumh, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

} // namespace glass

#endif
/* =================================================================================================== */
/*                                        simd/distance.hpp                                            */
#include <cstdint>
#include <cstdio>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace glass {

template <typename T1, typename T2, typename U, typename... Params>
using Dist = U (*)(const T1 *, const T2 *, int, Params...);

GLASS_INLINE inline void prefetch_L1(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

GLASS_INLINE inline void prefetch_L2(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

GLASS_INLINE inline void prefetch_L3(const void *address) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

inline void mem_prefetch(char *ptr, const int num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 28:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 27:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 26:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 25:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 24:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 23:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 22:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 21:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 20:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_L1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

FAST_BEGIN
inline float L2SqrRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
}
FAST_END

FAST_BEGIN
inline float IPRef(const float *x, const float *y, int d) {
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
}
FAST_END

inline float L2Sqr(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    auto t = _mm512_sub_ps(xx, yy);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(t, t));
  }
  return reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    auto t = vsubq_f32(xx, yy);
    sum = vmlaq_f32(sum, t, t);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0f;
  for (int i = 0; i < d; ++i) {
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return sum;
#endif
}

inline float IP(const float *x, const float *y, int d) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm512_loadu_ps(x);
    x += 16;
    auto yy = _mm512_loadu_ps(y);
    y += 16;
    sum = _mm512_add_ps(sum, _mm512_mul_ps(xx, yy));
  }
  return -reduce_add_f32x16(sum);
#elif defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  const float *end = x + d;
  while (x < end) {
    auto xx = _mm256_loadu_ps(x);
    x += 8;
    auto yy = _mm256_loadu_ps(y);
    y += 8;
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
  }
  return -reduce_add_f32x8(sum);
#elif defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0);
  for (int32_t i = 0; i < d; i += 4) {
    auto xx = vld1q_f32(x + i);
    auto yy = vld1q_f32(y + i);
    sum = vmlaq_f32(sum, xx, yy);
  }
  return vaddvq_f32(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    sum += x[i] * y[i];
  }
  return -sum;
#endif
}

inline float L2SqrSQ8_ext(const float *x, const uint8_t *y, int d,
                          const float *mi, const float *dif) {
#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    auto d = _mm512_sub_ps(_mm512_mul_ps(xx, const_255), yy);
    sum = _mm512_fmadd_ps(d, d, sum);
  }
  return reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = (y[i] + 0.5f);
    yy = yy * dif[i] + mi[i] * 255.0f;
    auto dif = x[i] * 255.0f - yy;
    sum += dif * dif;
  }
  return sum;
#endif
}

inline float IPSQ8_ext(const float *x, const uint8_t *y, int d, const float *mi,
                       const float *dif) {

#if defined(__AVX512F__)
  __m512 sum = _mm512_setzero_ps();
  __m512 dot5 = _mm512_set1_ps(0.5f);
  __m512 const_255 = _mm512_set1_ps(255.0f);
  for (int i = 0; i < d; i += 16) {
    auto zz = _mm_loadu_epi8(y + i);
    auto zzz = _mm512_cvtepu8_epi32(zz);
    auto yy = _mm512_cvtepi32_ps(zzz);
    yy = _mm512_add_ps(yy, dot5);
    auto mi512 = _mm512_loadu_ps(mi + i);
    auto dif512 = _mm512_loadu_ps(dif + i);
    yy = _mm512_mul_ps(yy, dif512);
    yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_255));
    auto xx = _mm512_loadu_ps(x + i);
    sum = _mm512_fmadd_ps(xx, yy, sum);
  }
  return -reduce_add_f32x16(sum);
#else
  float sum = 0.0;
  for (int i = 0; i < d; ++i) {
    float yy = y[i] + 0.5f;
    yy = yy * dif[i] + mi[i] * 255.0f;
    sum += x[i] * yy;
  }
  return -sum;
#endif
}

inline int32_t L2SqrSQ4(const uint8_t *x, const uint8_t *y, int d) {
#if defined(__AVX2__)
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  __m256i mask = _mm256_set1_epi8(0xf);
  for (int i = 0; i < d; i += 64) {
    auto xx = _mm256_loadu_si256((__m256i *)(x + i / 2));
    auto yy = _mm256_loadu_si256((__m256i *)(y + i / 2));
    auto xx1 = _mm256_and_si256(xx, mask);
    auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
    auto yy1 = _mm256_and_si256(yy, mask);
    auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
    auto d1 = _mm256_sub_epi8(xx1, yy1);
    auto d2 = _mm256_sub_epi8(xx2, yy2);
    d1 = _mm256_abs_epi8(d1);
    d2 = _mm256_abs_epi8(d2);
    sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
    sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i16x16(sum1);
#else
  int32_t sum = 0;
  for (int i = 0; i < d; ++i) {
    {
      int32_t xx = x[i / 2] & 15;
      int32_t yy = y[i / 2] & 15;
      sum += (xx - yy) * (xx - yy);
    }
    {
      int32_t xx = x[i / 2] >> 4 & 15;
      int32_t yy = y[i / 2] >> 4 & 15;
      sum += (xx - yy) * (xx - yy);
    }
  }
  return sum;
#endif
}

} // namespace glass
/* =================================================================================================== */
/*                                        quant/fp32_quant.hpp                                         */
namespace glass {

template <Metric metric, int DIM = 0> struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

  ~FP32Quantizer() { free(codes); }

  void train(const float *data, int64_t n) {
    codes = (char *)alloc2M(n * code_size);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  void encode(const float *from, char *to) { std::memcpy(to, from, d * 4); }

  char *get_data(int u) const { return codes + u * code_size; }

  template <typename Pool>
  void reorder(const Pool &pool, const float *, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func = metric == Metric::L2 ? L2Sqr : IP;
    const FP32Quantizer &quant;
    float *q = nullptr;
    Computer(const FP32Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
/* =================================================================================================== */
/*                                        quant/sq4_quant.hpp                                          */
#include <cmath>

namespace glass {

template <Metric metric, typename Reorderer = FP32Quantizer<metric>,
          int DIM = 0>
struct SQ4Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 128;
  float mx = -HUGE_VALF, mi = HUGE_VALF, dif;
  int d, d_align;
  int64_t code_size;
  data_type *codes = nullptr;

  Reorderer reorderer;

  SQ4Quantizer() = default;

  explicit SQ4Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align / 2),
        reorderer(dim) {}

  ~SQ4Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n * d; ++i) {
      mx = std::max(mx, data[i]);
      mi = std::min(mi, data[i]);
    }
    dif = mx - mi;
    codes = (data_type *)alloc2M(n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
    reorderer.train(data, n);
  }

  char *get_data(int u) const { return (char *)codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi) / dif;
      if (x < 0.0) {
        x = 0.0;
      }
      if (x > 0.999) {
        x = 0.999;
      }
      uint8_t y = 16 * x;
      if (j & 1) {
        to[j / 2] |= y << 4;
      } else {
        to[j / 2] |= y;
      }
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float *q, int *dst, int k) const {
    int cap = pool.capacity();
    auto computer = reorderer.get_computer(q);
    searcher::MaxHeap<typename Reorderer::template Computer<0>::dist_type> heap(
        k);
    for (int i = 0; i < cap; ++i) {
      if (i + 1 < cap) {
        computer.prefetch(pool.id(i + 1), 1);
      }
      int id = pool.id(i);
      float dist = computer(id);
      heap.push(id, dist);
    }
    for (int i = 0; i < k; ++i) {
      dst[i] = heap.pop();
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = int32_t;
    constexpr static auto dist_func = L2SqrSQ4;
    const SQ4Quantizer &quant;
    uint8_t *q;
    Computer(const SQ4Quantizer &quant, const float *query)
        : quant(quant), q((uint8_t *)alloc64B(quant.code_size)) {
      quant.encode(query, (char *)q);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
/* =================================================================================================== */
/*                                         quant/sq8_quant.hpp                                         */
#include <cmath>
#include <vector>

namespace glass {

template <Metric metric, int DIM = 0> struct SQ8Quantizer {
  using data_type = uint8_t;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char *codes = nullptr;
  std::vector<float> mx, mi, dif;

  SQ8Quantizer() = default;

  explicit SQ8Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align),
        mx(d_align, -HUGE_VALF), mi(d_align, HUGE_VALF), dif(d_align) {}

  ~SQ8Quantizer() { free(codes); }

  void train(const float *data, int n) {
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < d; ++j) {
        mx[j] = std::max(mx[j], data[i * d + j]);
        mi[j] = std::min(mi[j], data[i * d + j]);
      }
    }
    for (int64_t j = 0; j < d; ++j) {
      dif[j] = mx[j] - mi[j];
    }
    for (int64_t j = d; j < d_align; ++j) {
      dif[j] = mx[j] = mi[j] = 0;
    }
    codes = (char *)alloc2M((size_t)n * code_size);
    for (int i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  char *get_data(int u) const { return codes + u * code_size; }

  void encode(const float *from, char *to) const {
    for (int j = 0; j < d; ++j) {
      float x = (from[j] - mi[j]) / dif[j];
      if (x < 0) {
        x = 0.0;
      }
      if (x > 1.0) {
        x = 1.0;
      }
      uint8_t y = x * 255;
      to[j] = y;
    }
  }

  template <typename Pool>
  void reorder(const Pool &pool, const float * /**q*/, int *dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  template <int DALIGN = do_align(DIM, kAlign)> struct Computer {
    using dist_type = float;
    constexpr static auto dist_func =
        metric == Metric::L2 ? L2SqrSQ8_ext : IPSQ8_ext;
    const SQ8Quantizer &quant;
    float *q;
    const float *mi, *dif;
    Computer(const SQ8Quantizer &quant, const float *query)
        : quant(quant), q((float *)alloc64B(quant.d_align * 4)),
          mi(quant.mi.data()), dif(quant.dif.data()) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const {
      return dist_func(q, (data_type *)quant.get_data(u), quant.d_align, mi,
                       dif);
    }
    void prefetch(int u, int lines) const {
      mem_prefetch(quant.get_data(u), lines);
    }
  };

  auto get_computer(const float *query) const {
    return Computer<0>(*this, query);
  }
};

} // namespace glass
/* =================================================================================================== */
/*                                          quant/quant.hpp                                            */
#include <string>
#include <unordered_map>

namespace glass {

enum class QuantizerType { FP32, SQ8, SQ4 };

inline std::unordered_map<int, QuantizerType> quantizer_map;

inline int quantizer_map_init = [] {
  quantizer_map[0] = QuantizerType::FP32;
  quantizer_map[1] = QuantizerType::SQ8;
  quantizer_map[2] = QuantizerType::SQ8;
  return 42;
}();

} // namespace glass
/* =================================================================================================== */
/*                                             graph.hpp                                               */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

namespace glass {

constexpr int EMPTY_ID = -1;

template <typename node_t> struct Graph {
  int N;                // 节点个数
  int K;                // 每个节点的邻居节点的个数上限

  node_t *data = nullptr;                   // N * K 大小的邻接列表

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t *edges, int N, int K) : N(N), K(K), data(edges) {}

  Graph(int N, int K)
      : N(N), K(K), data((node_t *)alloc2M((size_t)N * K * sizeof(node_t))) {}

  Graph(const Graph &g) : Graph(g.N, g.K) {
    this->eps = g.eps;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        at(i, j) = g.at(i, j);
      }
    }
  }

  void init(int N, int K) {
    data = (node_t *)alloc2M((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  // 获取编号 u 的数据的所有出边
  const int *edges(int u) const { return data + K * u; }
  int *edges(int u) { return data + K * u; }

  // 获取编号 i 的数据的第 j 个邻居节点
  node_t at(int i, int j) const { return data[i * K + j]; }
  node_t &at(int i, int j) { return data[i * K + j]; }

  void prefetch(int u, int lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    for (auto ep : eps) {
      pool.insert(ep, computer(ep));
      pool.vis.set(ep);
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same<node_t, int32_t>::value);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Graph Saving done\n");
#endif
  }

  void load(const std::string &filename) {
    static_assert(std::is_same<node_t, int32_t>::value);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)alloc2M((size_t)N * K * 4);
    reader.read((char *)data, N * K * 4);
    if (reader.peek() != EOF) {
      ;
    }
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Graph Loding done\n");
#endif
  }
};

} // namespace glass
/* =================================================================================================== */
/*                                             build.hpp                                               */
namespace glass {

struct Builder {
  virtual void Build(float *data, int nb) = 0;
  virtual glass::Graph<int> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace glass
/* =================================================================================================== */
/*                                           searcher.hpp                                              */
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace glass {

struct SearcherBase {
  virtual void SetData(const float *data, int n, int dim) = 0;
  virtual void Optimize() = 0;
  virtual void Search(const float *q, int k, int *dst) const = 0;
  virtual void SetEf(int ef) = 0;
  virtual ~SearcherBase() = default;
};

template <typename Quantizer> struct Searcher : public SearcherBase {

  int d;
  int nb;
  Graph<int> graph;
  Quantizer quant;

  // Search parameters
  int ef = 32;

  // Memory prefetch parameters
  int po = 1;
  int pl = 1;

  // Optimization parameters
  constexpr static int kOptimizePoints = 1000;
  constexpr static int kTryPos = 10;
  constexpr static int kTryPls = 5;
  constexpr static int kTryK = 10;
  int sample_points_num;
  std::vector<float> optimize_queries;
  const int graph_po;

  Searcher(const Graph<int> &graph) : graph(graph), graph_po(graph.K / 16) {}

  void SetData(const float *data, int n, int dim) override {
    this->nb = n;
    this->d = dim;
    quant = Quantizer(d);
    quant.train(data, n);

    sample_points_num = std::min(kOptimizePoints, nb - 1);
    std::vector<int> sample_points(sample_points_num);
    std::mt19937 rng;
    GenRandom(rng, sample_points.data(), sample_points_num, nb);
    optimize_queries.resize(sample_points_num * d);
    for (int i = 0; i < sample_points_num; ++i) {
      memcpy(optimize_queries.data() + i * d, data + sample_points[i] * d,
             d * sizeof(float));
    }
  }

  void SetEf(int ef) override { this->ef = ef; }

  void Optimize() override {
    std::vector<int> try_pos(std::min(kTryPos, graph.K));
    std::vector<int> try_pls(
        std::min(kTryPls, (int)upper_div(quant.code_size, 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);
    std::vector<int> dummy_dst(kTryK);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============Start optimization=============\n");
#endif
    { // warmup
      for (int i = 0; i < sample_points_num; ++i) {
        Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
      }
    }

    float min_ela = std::numeric_limits<float>::max();
    int best_po = 0, best_pl = 0;
    for (auto try_po : try_pos) {
      for (auto try_pl : try_pls) {
        this->po = try_po;
        this->pl = try_pl;
        auto st = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < sample_points_num; ++i) {
          Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
        }

        auto ed = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(ed - st).count();
        if (ela < min_ela) {
          min_ela = ela;
          best_po = try_po;
          best_pl = try_pl;
        }
      }
    }
    this->po = 1;
    this->pl = 1;
#ifdef DEBUG_ENABLED
    auto st = std::chrono::high_resolution_clock::now();
#endif
    for (int i = 0; i < sample_points_num; ++i) {
      Search(optimize_queries.data() + i * d, kTryK, dummy_dst.data());
    }
#ifdef DEBUG_ENABLED
    auto ed = std::chrono::high_resolution_clock::now();
    float baseline_ela = std::chrono::duration<double>(ed - st).count();
    fprintf(stderr, "settint best po = %d, best pl = %d\n"
           "gaining %.2f%% performance improvement\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1));
#endif
    this->po = best_po;
    this->pl = best_pl;
  }

  void Search(const float *q, int k, int *dst) const override {
    auto computer = quant.get_computer(q);
    searcher::LinearPool<typename Quantizer::template Computer<0>::dist_type> pool(nb, std::max(k, ef), k);
    // searcher::HeapPool<typename Quantizer::template Computer<0>::dist_type> pool(nb, std::max(k, ef), k);
    graph.initialize_search(pool, computer);
    SearchImpl(pool, computer);
    quant.reorder(pool, q, dst, k);
  }

  // pool 存储 k 个最近邻节点的候选节点，其大小(记为 l )会大于 k
  // 每次都从 pool 中取出第一个还没有被访问过的节点
  // 将该节点标记为已访问，将其所有邻居节点都加入 pool
  // 计算 pool 中所有节点距离要查询的数据 q 的距离， 按照距离的升序对 pool 中的节点进行排序
  // 如果此时 pool 的元素个数超过了 l， 将多余的节点删除
  template <typename Pool, typename Computer>
  void SearchImpl(Pool &pool, const Computer &computer) const {
#ifdef DEBUG_ENABLED
    int compute_num = 0;
#endif
    while (pool.has_next()) {
      auto u = pool.pop();
      graph.prefetch(u, graph_po);
      for (int i = 0; i < po; ++i) {
        int to = graph.at(u, i);
        computer.prefetch(to, pl);
      }
      for (int i = 0; i < graph.K; ++i) {
        int v = graph.at(u, i);
        if (v == EMPTY_ID) {
          break;
        }
        if (pool.vis.get(v)) {
          continue;
        }
        pool.vis.set(v);
        if (i + po < graph.K && graph.at(u, i + po) != -1) {
          int to = graph.at(u, i + po);
          computer.prefetch(to, pl);
        }
        auto cur_dist = computer(v);
        pool.insert(v, cur_dist);
#ifdef DEBUG_ENABLED
        compute_num ++;
#endif
      }
    }
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Compute number: %d\n", compute_num);
#endif
  }
};

inline std::unique_ptr<SearcherBase> create_searcher(const Graph<int> &graph,
                                                     const std::string &metric,
                                                     int level = 1) {
  auto m = metric_map[metric];
  if (level == 0) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<FP32Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else if (level == 1) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ8Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else if (level == 2) {
    if (m == Metric::L2) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::L2>>>(graph);
    } else if (m == Metric::IP) {
      return std::make_unique<Searcher<SQ4Quantizer<Metric::IP>>>(graph);
    } else {
#ifdef DEBUG_ENABLED
      fprintf(stderr, "Metric not suppported\n");
#endif
      return nullptr;
    }
  } else {
#ifdef DEBUG_ENABLED
    fprintf(stderr, "Quantizer type not supported\n");
#endif
    return nullptr;
  }
}

} // namespace glass
/* =================================================================================================== */
/*                                          nsg/nndescent.hpp                                          */
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace glass {

// 使用 NNDesent 算法构建出 KNNG 
struct NNDescent {
  // 每一个节点的若干个近似最近邻节点的信息
  struct Nhood {
    std::vector<Neighbor> pool; // candidate pool (a max heap) ，里面存储的是近似的最近邻节点，按照邻居到该节点的距离组织成大顶堆的形式
    int M;                      // pool 的大小
    std::vector<int> nn_new, nn_old, rnn_new, rnn_old;        // 这四个vector的大小是 2 * M，并且里面存储的都是不重复的数据的编号，并且按照升序排列

    Nhood(std::mt19937 &rng, int s, int64_t N) {
      M = s;
      nn_new.resize(s * 2);
      GenRandom(rng, nn_new.data(), (int)nn_new.size(), N);
    }

    Nhood &operator=(const Nhood &other) {
      M = other.M;
      std::copy(other.nn_new.begin(), other.nn_new.end(),
                std::back_inserter(nn_new));
      nn_new.reserve(other.nn_new.capacity());
      pool.reserve(other.pool.capacity());
      return *this;
    }

    Nhood(const Nhood &other) {
      M = other.M;
      std::copy(other.nn_new.begin(), other.nn_new.end(),
                std::back_inserter(nn_new));
      nn_new.reserve(other.nn_new.capacity());
      pool.reserve(other.pool.capacity());
    }

    // 尝试将编号为 id ，与当前节点距离为 dist 的节点插入当前节点的 pool 中
    void insert(int id, float dist) {
      if (dist > pool.front().distance)
        return;
      for (int i = 0; i < (int)pool.size(); i++) {
        if (id == pool[i].id)
          return;
      }
      if (pool.size() < pool.capacity()) {
        pool.push_back(Neighbor(id, dist, true));
        std::push_heap(pool.begin(), pool.end());
      } else {
        std::pop_heap(pool.begin(), pool.end());
        pool[pool.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(pool.begin(), pool.end());
      }
    }

    template <typename C> void join(C callback) const {
      for (int const i : nn_new) {
        for (int const j : nn_new) {
          if (i < j) {
            callback(i, j);
          }
        }
        for (int j : nn_old) {
          callback(i, j);
        }
      }
    }
  };

  std::vector<Nhood> graph;             // 存储 nb 个节点的邻居信息
  Graph<int> final_graph;               // 最终构建出来的 KNNG
  int64_t d;                            // 数据维度
  int64_t nb;                           // 数据个数
  const float *data;                    // nb * d 大小的数据集
  int K;                                // 最近邻居的数量
  int S = 10;                           // 每个节点的 pool 初始的时候存储的邻居数量
  int R = 100;
  int iters = 10;                       // 迭代次数
  int random_seed = 347;
  int L;                                // 每个节点的 pool 的存储的邻居的数量上限
  Dist<float, float, float> dist_func;

  NNDescent(int64_t dim, const std::string &metric) : d(dim) {
    if (metric == "L2") {
      dist_func = L2SqrRef;
    } else if (metric == "IP") {
      dist_func = IPRef;
    }
  }

  // 根据 n 个数据量的数据集 data ，构建最近邻居数量为 K 的 final_graph
  void Build(const float *data, int n, int K) {
    this->data = data;
    this->nb = n;
    this->K = K;
    this->L = K + 50;
    Init();
    Descent();
    final_graph.init(n, K);
    for (int i = 0; i < nb; i++) {
      std::sort(graph[i].pool.begin(), graph[i].pool.end());
      for (int j = 0; j < K; j++) {
        final_graph.at(i, j) = graph[i].pool[j].id;
      }
    }
    std::vector<Nhood>().swap(graph);
  }

  void Init() {
    // 随机初始化所有节点的 2 * S 个 nn_new
    // 初始化的时候所有节点的 nn_new 都是随机的、不重复的、并且编号是递增的
    graph.reserve(nb);
    {
      std::mt19937 rng(random_seed * 6007);
      for (int i = 0; i < nb; ++i) {
        graph.emplace_back(rng, S, nb);
      }
    }
    {
      std::mt19937 rng(random_seed * 7741);
      // 随机初始化所有节点的 pool
      // 初始化完成之后的 pool 存储的邻居数量上限是 L ，目前装有 S 或者 S - 1 个随机生成的邻居（不包含节点自身）
      // 并将每个节点的 pool 按照其中邻居到该节点的距离组织成一个大顶堆
      for (int i = 0; i < nb; ++i) {
        std::vector<int> tmp(S);
        GenRandom(rng, tmp.data(), S, nb);
        for (int j = 0; j < S; j++) {
          int id = tmp[j];
          if (id == i)
            continue;
          float dist = dist_func(data + i * d, data + id * d, d);
          graph[i].pool.push_back(Neighbor(id, dist, true));
        }
        std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
        graph[i].pool.reserve(L);
      }
    }
  }

  void Descent() {
    int num_eval = std::min((int64_t)100, nb);
#ifdef DEBUG_ENABLED
    std::mt19937 rng(random_seed * 6577);
    std::vector<int> eval_points(num_eval);
    std::vector<std::vector<int>> eval_gt(num_eval);
    GenRandom(rng, eval_points.data(), num_eval, nb);
    GenEvalGt(eval_points, eval_gt);
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    for (int iter = 1; iter <= iters; ++iter) {
      Join();
      Update();
#ifdef DEBUG_ENABLED
      float recall = EvalRecall(eval_points, eval_gt);
      fprintf(stderr, "NNDescent iter: [%d/%d], recall: %f\n", iter, iters, recall);
#endif
    }
#ifdef DEBUG_ENABLED
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "NNDescent cost: %.2lfs\n", ela);
#endif
  }

  // 更新所有节点的 pool
  void Join() {
    // 遍历所有的节点，计算每一个节点的 nn_new 内部的所有节点的两两组合以及 nn_new 和 nn_old 之间的节点的两两组合的距离，并且尝试更新这些节点的 pool
    for (int u = 0; u < nb; u++) {
      graph[u].join([&](int i, int j) {
        if (i != j) {
          float dist = dist_func(data + i * d, data + j * d, d);
          graph[i].insert(j, dist);
          graph[j].insert(i, dist);
        }
      });
    }
  }

  void Update() {
    // 清空所有节点的 nn_new 和 nn_old
    for (int i = 0; i < nb; i++) {
      std::vector<int>().swap(graph[i].nn_new);               // 这种技巧通常被用来在清空一个 std::vector 对象时，同时释放它所占用的内存空间，而不需要重新分配新的内存空间
      std::vector<int>().swap(graph[i].nn_old);
    }
    // 对每一个节点的 pool 根据距离进行升序排序，并且保证 pool 存储的邻居数量都不超过 L
    // 以某种方式更新每一个节点的 M
    for (int n = 0; n < nb; ++n) {
      auto &nn = graph[n];
      std::sort(nn.pool.begin(), nn.pool.end());
      if ((int)nn.pool.size() > L) {
        nn.pool.resize(L);
      }
      nn.pool.reserve(L);
      int maxl = std::min(nn.M + S, (int)nn.pool.size());
      int c = 0;
      int l = 0;
      while ((l < maxl) && (c < S)) {
        if (nn.pool[l].flag)
          ++c;
        ++l;
      }
      nn.M = l;
    }
    {
      std::mt19937 rng(random_seed * 5081);
      for (int n = 0; n < nb; ++n) {
        auto &node = graph[n];
        auto &nn_new = node.nn_new;
        auto &nn_old = node.nn_old;
        for (int l = 0; l < node.M; ++l) {
          auto &nn = node.pool[l];
          auto &other = graph[nn.id];
          if (nn.flag) {
            nn_new.push_back(nn.id);
            if (nn.distance > other.pool.back().distance) {
              if ((int)other.rnn_new.size() < R) {
                other.rnn_new.push_back(n);
              } else {
                int pos = rng() % R;
                other.rnn_new[pos] = n;
              }
            }
            nn.flag = false;
          } else {
            nn_old.push_back(nn.id);
            if (nn.distance > other.pool.back().distance) {
              if ((int)other.rnn_old.size() < R) {
                other.rnn_old.push_back(n);
              } else {
                int pos = rng() % R;
                other.rnn_old[pos] = n;
              }
            }
          }
        }
        std::make_heap(node.pool.begin(), node.pool.end());
      }
    }
    // 将每个节点的 rnn_new 插入到 nn_new 中， rnn_old 插入到 nn_old 中，然后将 rnn_new 和 rnn_old 清空
    // 确保每个节点的 nn_old 存储的节点数量不超过 2 * R
    for (int i = 0; i < nb; ++i) {
      auto &nn_new = graph[i].nn_new;
      auto &nn_old = graph[i].nn_old;
      auto &rnn_new = graph[i].rnn_new;
      auto &rnn_old = graph[i].rnn_old;
      nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
      nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
      if ((int)nn_old.size() > R * 2) {
        nn_old.resize(R * 2);
        nn_old.reserve(R * 2);
      }
      std::vector<int>().swap(graph[i].rnn_new);
      std::vector<int>().swap(graph[i].rnn_old);
    }
  }

  void GenEvalGt(const std::vector<int> &eval_set,
                 std::vector<std::vector<int>> &eval_gt) {
    for (int i = 0; i < (int)eval_set.size(); i++) {
      std::vector<Neighbor> tmp;
      for (int j = 0; j < nb; j++) {
        if (eval_set[i] == j)
          continue;
        float dist = dist_func(data + eval_set[i] * d, data + j * d, d);
        tmp.push_back(Neighbor(j, dist, true));
      }
      std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
      for (int j = 0; j < K; j++) {
        eval_gt[i].push_back(tmp[j].id);
      }
    }
  }

  float EvalRecall(const std::vector<int> &eval_set,
                   const std::vector<std::vector<int>> &eval_gt) {
    float mean_acc = 0.0f;
    for (int i = 0; i < (int)eval_set.size(); i++) {
      float acc = 0;
      std::vector<Neighbor> &g = graph[eval_set[i]].pool;
      const std::vector<int> &v = eval_gt[i];
      for (int j = 0; j < (int)g.size(); j++) {
        for (int k = 0; k < (int)v.size(); k++) {
          if (g[j].id == v[k]) {
            acc++;
            break;
          }
        }
      }
      mean_acc += acc / v.size();
    }
    return mean_acc / eval_set.size();
  }
};

} // namespace glass
/* =================================================================================================== */
/*                                            nsg/nsg.hpp                                              */
#include <random>
#include <stack>

namespace glass {

struct NSG : public Builder {
  int d;            // 数据维度
  std::string metric;         // 数据之间的度量方式
  int R;            // 构建出来的 NSG 的每个节点的邻居节点数量的上限
  int L;            // 在进行图搜索的时候候选池的大小（这里的图搜索只是NSG构建过程中的图搜索，不是查询的时候）
  int C;
  int nb;           // 数据个数
  float *data;      // 数据集
  int ep;           // Navigating Node，即使用 NSG 进行搜索时的起点
  Graph<int> final_graph;             // 最终构建出来的 NSG
  RandomGenerator rng; ///< random generator
  Dist<float, float, float> dist_func;
  int GK;
  int nndescent_S;
  int nndescent_R;
  int nndescent_L;
  int nndescent_iter;

  explicit NSG(int dim, const std::string &metric, int R = 32, int L = 200)
      : d(dim), metric(metric), R(R), L(L), rng(0x0903) {
    this->C = R + 100;
    srand(0x1998);
    if (metric == "L2") {
      dist_func = L2SqrRef;
    } else if (metric == "IP") {
      dist_func = IPRef;
    }
    this->GK = 64;
    // this->GK = 128;
    this->nndescent_S = 10;
    this->nndescent_R = 100;
    this->nndescent_L = this->GK + 50;
    // this->nndescent_iter = 10;
    this->nndescent_iter = 7;
  }

  void Build(float *data, int n) override {
    this->nb = n;
    this->data = data;
    NNDescent nnd(d, metric);
    nnd.S = nndescent_S;
    nnd.R = nndescent_R;
    nnd.L = nndescent_L;
    nnd.iters = nndescent_iter;
    // 创建 KNN Graph（大概的）
    nnd.Build(data, n, GK);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============NNDescent parameters=============\n");
    fprintf(stderr, "GK: %d\n", GK);
    fprintf(stderr, "S: %d\n", nndescent_S);
    fprintf(stderr, "R: %d\n", nndescent_R);
    fprintf(stderr, "L: %d\n", nndescent_L);
    fprintf(stderr, "iters: %d\n", nndescent_iter);
#endif
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============NSG parameters=============\n");
    fprintf(stderr, "R: %d\n", this->R);
    fprintf(stderr, "L: %d\n", this->L);
    fprintf(stderr, "C: %d\n", this->C);
    fprintf(stderr, "GK: %d\n", this->GK);
#endif
    const auto &knng = nnd.final_graph;
    // 初始化 KNNG， 获得 Navigating Node
    Init(knng);
    std::vector<int> degrees(n, 0);
    {
      Graph<Node> tmp_graph(n, R);
      // 构建出 NSG tmp_graph
      link(knng, tmp_graph);
      // 把 tmp_graph 复制一份给 final_graph，就是最终构建出来的 NSG
      // 统计出所有节点的出度 degrees
      final_graph.init(n, R);
      std::fill_n(final_graph.data, n * R, EMPTY_ID);
      final_graph.eps = {ep};
      for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int j = 0; j < R; j++) {
          int id = tmp_graph.at(i, j).id;
          if (id != EMPTY_ID) {
            final_graph.at(i, cnt) = id;
            cnt += 1;
          }
        }
        degrees[i] = cnt;
      }
    }
    // 把那些不在 final_graph 中的节点加入到 final_graph 中，保证连通性
    int num_attached = tree_grow(degrees);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "num_attached: %d\n", num_attached);
    int max = 0, min = 1e6;
    double avg = 0;
    for (int i = 0; i < n; i++) {
      int size = 0;
      while (size < R && final_graph.at(i, size) != EMPTY_ID) {
        size += 1;
      }
      max = std::max(size, max);
      min = std::min(size, min);
      avg += size;
    }
    avg = avg / n;
    fprintf(stderr, "Degree Statistics: Max = %d, Min = %d, Avg = %lf\n", max, min, avg);
#endif
  }

  Graph<int> GetGraph() override { return final_graph; }

  void Init(const Graph<int> &knng) {
    // 计算出数据集的重心 center（这个重心大概率不在数据集中）
    std::vector<float> center(d);
    for (int i = 0; i < d; ++i) {
      center[i] = 0.0;
    }
    for (int i = 0; i < nb; i++) {
      for (int j = 0; j < d; j++) {
        center[j] += data[i * d + j];
      }
    }
    for (int i = 0; i < d; i++) {
      center[i] /= nb;
    }
    int ep_init = rng.rand_int(nb);
    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;
    std::vector<bool> vis(nb);
    // 以 center 为查询点，在 knng 中搜索最近邻点，将最近邻点作为 Navigating Node
    search_on_graph<false>(center.data(), knng, vis, ep_init, L, retset, tmpset);
    // set enterpoint
    this->ep = retset[0].id;
  }

  // 在图 graph 中对数据 q 进行图搜索
  // vis 用来标记图中的节点是否被访问过，ep 是图搜索的起点，pool_size 是搜索的池子的大小
  // retset 是返回值，存储着图中距离数据 q 最近的 k 个（k就是 graph 的 K）邻居节点
  // fullset 用来存储搜索路径上遇到的所有节点以及这些节点到 q 的距离
  template <bool collect_fullset>
  void search_on_graph(const float *q, const Graph<int> &graph,
                       std::vector<bool> &vis, int ep, int pool_size,
                       std::vector<Neighbor> &retset,
                       std::vector<Node> &fullset) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);
    int num_ids = 0;
    for (int i = 0; i < (int)init_ids.size() && i < graph.K; i++) {
      int id = (int)graph.at(ep, i);
      if (id < 0 || id >= nb) {
        continue;
      }
      init_ids[i] = id;
      vis[id] = true;
      num_ids += 1;
    }
    while (num_ids < pool_size) {
      int id = gen.rand_int(nb);
      if (vis[id]) {
        continue;
      }
      init_ids[num_ids] = id;
      num_ids++;
      vis[id] = true;
    }
    for (int i = 0; i < (int)init_ids.size(); i++) {
      int id = init_ids[i];
      float dist = dist_func(q, data + id * d, d);
      retset[i] = Neighbor(id, dist, true);
      if (collect_fullset) {
        fullset.emplace_back(retset[i].id, retset[i].distance);
      }
    }
    std::sort(retset.begin(), retset.begin() + pool_size);
    int k = 0;
    while (k < pool_size) {
      int updated_pos = pool_size;
      if (retset[k].flag) {
        retset[k].flag = false;
        int n = retset[k].id;
        for (int m = 0; m < graph.K; m++) {
          int id = (int)graph.at(n, m);
          if (id < 0 || id > nb || vis[id]) {
            continue;
          }
          vis[id] = true;
          float dist = dist_func(q, data + id * d, d);
          Neighbor nn(id, dist, true);
          if (collect_fullset) {
            fullset.emplace_back(id, dist);
          }
          if (dist >= retset[pool_size - 1].distance) {
            continue;
          }
          int r = insert_into_pool(retset.data(), pool_size, nn);
          updated_pos = std::min(updated_pos, r);
        }
      }
      k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
  }

  // 根据 KNN Graph knng 构建 NSG graph
  void link(const Graph<int> &knng, Graph<Node> &graph) {
#ifdef DEBUG_ENABLED
    auto st = std::chrono::high_resolution_clock::now();
    int cnt = 0;
#endif
    // 遍历所有节点
    for (int i = 0; i < nb; i++) {
      std::vector<Node> pool;                     // 图搜索过程中访问过的所有节点以及它们到当前节点 i 的距离
      std::vector<Neighbor> tmp;                  // 图搜索得到的当前节点 i 的最近邻节点
      std::vector<bool> vis(nb);                  // 图搜索过程中所有访问过的节点
      // 从 ep 开始，对当前节点 i 进行在 knng 中进行图搜索
      search_on_graph<true>(data + i * d, knng, vis, ep, L, tmp, pool);
      // 根据 knng，使用 MRNG （Monotonic Relative Neighborhood Graph）的边选择策略把 graph 构建成 NSG
      sync_prune(i, pool, vis, knng, graph);
      pool.clear();
      tmp.clear();
#ifdef DEBUG_ENABLED
      int cur = cnt += 1;
      if (cur % 10000 == 0) {
        fprintf(stderr, "NSG building progress: [%d/%d]\n", cur, nb);
      }
#endif
    }
#ifdef DEBUG_ENABLED
    auto ed = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(ed - st).count();
    fprintf(stderr, "NSG building cost: %.2lfs\n", ela);
#endif

    for (int i = 0; i < nb; ++i) {
      add_reverse_links(i, graph);
    }
  }

  void sync_prune(int q, std::vector<Node> &pool, std::vector<bool> &vis,
                  const Graph<int> &knng, Graph<Node> &graph) {
    for (int i = 0; i < knng.K; i++) {
      int id = knng.at(q, i);
      if (id < 0 || id >= nb || vis[id]) {
        continue;
      }

      float dist = dist_func(data + q * d, data + id * d, d);
      pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
      start++;
    }
    result.push_back(pool[start]);

    while ((int)result.size() < R && (++start) < (int)pool.size() && start < C) {
      auto &p = pool[start];
      bool occlude = false;
      for (int t = 0; t < (int)result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }

        float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
        if (djk < p.distance /* dik */) {
          occlude = true;
          break;
        }
      }
      if (!occlude) {
        result.push_back(p);
      }
    }

    for (int i = 0; i < R; i++) {
      if (i < (int)result.size()) {
        graph.at(q, i).id = result[i].id;
        graph.at(q, i).distance = result[i].distance;
      } else {
        graph.at(q, i).id = EMPTY_ID;
      }
    }
  }

  void add_reverse_links(int q, Graph<Node> &graph) {
    for (int i = 0; i < R; i++) {
      if (graph.at(q, i).id == EMPTY_ID) {
        break;
      }

      Node sn(q, graph.at(q, i).distance);
      int des = graph.at(q, i).id;

      std::vector<Node> tmp_pool;
      int dup = 0;
      {
        for (int j = 0; j < R; j++) {
          if (graph.at(des, j).id == EMPTY_ID) {
            break;
          }
          if (q == graph.at(des, j).id) {
            dup = 1;
            break;
          }
          tmp_pool.push_back(graph.at(des, j));
        }
      }

      if (dup) {
        continue;
      }

      tmp_pool.push_back(sn);
      if ((int)tmp_pool.size() > R) {
        std::vector<Node> result;
        int start = 0;
        std::sort(tmp_pool.begin(), tmp_pool.end());
        result.push_back(tmp_pool[start]);

        while ((int)result.size() < R && (++start) < (int)tmp_pool.size()) {
          auto &p = tmp_pool[start];
          bool occlude = false;
          for (int t = 0; t < (int)result.size(); t++) {
            if (p.id == result[t].id) {
              occlude = true;
              break;
            }
            float djk = dist_func(data + result[t].id * d, data + p.id * d, d);
            if (djk < p.distance /* dik */) {
              occlude = true;
              break;
            }
          }
          if (!occlude) {
            result.push_back(p);
          }
        }

        {
          for (int t = 0; t < (int)result.size(); t++) {
            graph.at(des, t) = result[t];
          }
        }

      } else {
        for (int t = 0; t < R; t++) {
          if (graph.at(des, t).id == EMPTY_ID) {
            graph.at(des, t) = sn;
            break;
          }
        }
      }
    }
  }

  // 通过不断深度优先遍历 final_graph，把数据集中不在 final_graph 中的点加入其中
  // 返回加入的节点的数量
  int tree_grow(std::vector<int> &degrees) {
    std::vector<bool> vis(nb);
    int num_attached = 0;
    int cnt = dfs(vis, ep, 0);
    while (true) {
      int sp = attach_unlinked(vis, degrees);
      if (sp == EMPTY_ID) break;
      num_attached ++;
    }
    return num_attached;
  }

  // 在 final_graph 中进行深度优先遍历
  // 遍历的起点是 root 点，vis 用来存储那些已经被访问过的节点，cnt 是遍历过的节点数量
  // 返回遍历的节点数量
  int dfs(std::vector<bool> &vis, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);
    if (vis[root]) {
      cnt++;
    }
    vis[root] = true;
    while (!stack.empty()) {
      int next = EMPTY_ID;
      for (int i = 0; i < R; i++) {
        int id = final_graph.at(node, i);
        if (id != EMPTY_ID && !vis[id]) {
          next = id;
          break;
        }
      }
      if (next == EMPTY_ID) {
        stack.pop();
        if (stack.empty()) {
          break;
        }
        node = stack.top();
        continue;
      }
      node = next;
      vis[node] = true;
      stack.push(node);
      cnt++;
    }
    return cnt;
  }

  // 取出一个不在 final_graph 中的点，记为 id ，再从 final_graph 中找出一个 node ，将边 (node, id) 加入到 final_graph 中
  // 并更新节点的出度数据 degrees 以及存在于 final_graph 中的点 vis
  // 返回 node
  int attach_unlinked(std::vector<bool> &vis, std::vector<int> &degrees) {
    // 找出一个还不在 final_graph 中的数据点 id
    int id = EMPTY_ID;
    for (int i = 0; i < nb; i++) {
      if (!vis[i]) {
        id = i;
        break;
      }
    }
    if (id == EMPTY_ID) {
      return EMPTY_ID;
    }
    std::vector<bool> vis2(nb);
    std::vector<Neighbor> tmp;
    std::vector<Node> pool;
    // 以 ep 为起点，id 为被查询数据在 final_graph 中进行图搜索
    search_on_graph<true>(data + id * d, final_graph, vis2, ep, L, tmp, pool);
    // 把图搜索过程中经过的所有点按照它们和 id 的顺序进行升序排序
    std::sort(pool.begin(), pool.end());
    int node = EMPTY_ID;
    bool found = false;
    // 从图搜素过程中经过的所有点中，找到那个出度小于 R 的，离 id 最近的那个点记为 node
    for (int i = 0; i < (int)pool.size(); i++) {
      node = pool[i].id;
      if (degrees[node] < R && node != id) {
        found = true;
        break;
      }
    }
    // 如果没有找到满足上述条件的点，那就随机找一个出度小于 R 的点记为 node
    if (!found) {
      do {
        node = rng.rand_int(nb);
        if (vis[node] && degrees[node] < R && node != id) {
          found = true;
        }
      } while (!found);
    }
    // 把边 (node, id) 加入到 final_graph 中
    int pos = degrees[node];
    final_graph.at(node, pos) = id;
    degrees[node] += 1;
    vis[id] = true;
    return node;
  }
};

} // namespace glass
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */
/* =================================================================================================== */

#define K_MAX 10
#define K_MIN 3

// Data Part
int N, D;
float *database;

// Distance function
float distance(const std::vector<float>& x, const std::vector<float>& p) {
    float sum = 0;
    for (int i = 0; i < D; ++i) {
        sum += (x[i] - p[i]) * (x[i] - p[i]);
    }
    return sqrt(sum);
}

// ==========================================================================================

std::unique_ptr<glass::SearcherBase> searcher;

// Preprocess Part
void preprocess() {
    // std::unique_ptr<glass::Builder> index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(D, "L2", 32, 50));
    std::unique_ptr<glass::Builder> index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(D, "L2", 32, 100));
    index->Build(database, N);
    glass::Graph<int> graph = index->GetGraph();
    int level = 0;
    int ef = 395;
    searcher = create_searcher(graph, "L2", level);
    searcher->SetData(database, N, D);
    searcher->SetEf(ef);
#ifdef DEBUG_ENABLED
    fprintf(stderr, "=============Search parameters=============\n");
    fprintf(stderr, "optimize level: %d\n", level);
    fprintf(stderr, "ef: %d\n", ef);
#endif
}

// ==========================================================================================


// ==========================================================================================

// Query Part
void query(const float *q, const int k, int *idxs) {
    searcher->Search(q, k, idxs);
}

// ==========================================================================================
int main() {
#ifdef DEBUG_ENABLED
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    scanf("%d", &N);
    scanf("%d", &D);
    database = new float[N * D];
    for (int i = 0; i < (N * D); i++) {
      scanf("%f", (database + i));
    }
#ifdef DEBUG_ENABLED
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    double ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "ReadIO cost: %.4lfs\n", ela);
#endif

#ifdef DEBUG_ENABLED
    t1 = std::chrono::high_resolution_clock::now();
#endif
    preprocess();
#ifdef DEBUG_ENABLED
    t2 = std::chrono::high_resolution_clock::now();
    ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "Preprocess cost: %.4lfs\n", ela);
#endif

    float *q = new float[D];

    printf("ok\n");
    fflush(stdout);

    int k;
    scanf("%d", &k);

    int *idxs = new int[k];

#ifdef DEBUG_ENABLED
    t1 = std::chrono::high_resolution_clock::now();
#endif
    while (true) {
#ifdef DEBUG_ENABLED
    auto read_t1 = std::chrono::high_resolution_clock::now();
#endif
        for (int i = 0; i < D; ++i) {
            if (scanf("%f", (q + i)) == 0) goto out;
        }
#ifdef DEBUG_ENABLED
    auto read_t2 = std::chrono::high_resolution_clock::now();
    auto read_ela = std::chrono::duration<double>(read_t2 - read_t1).count();
    fprintf(stderr, "Read query cost: %.4lfs\n", read_ela);
#endif
#ifdef DEBUG_ENABLED
    auto query_t1 = std::chrono::high_resolution_clock::now();
#endif
        query(q, k, idxs);
#ifdef DEBUG_ENABLED
    auto query_t2 = std::chrono::high_resolution_clock::now();
    auto query_ela = std::chrono::duration<double>(query_t2 - query_t1).count();
    fprintf(stderr, "Query cost: %.4lfs\n", query_ela);
#endif
#ifdef DEBUG_ENABLED
    auto write_t1 = std::chrono::high_resolution_clock::now();
#endif
        // Output k nearest indices
        for (int i = 0; i < k; i++) {
            printf("%d ", idxs[i]);
        }
        printf("\n");
        fflush(stdout);

#ifdef DEBUG_ENABLED
    auto write_t2 = std::chrono::high_resolution_clock::now();
    auto write_ela = std::chrono::duration<double>(write_t2 - write_t1).count();
    fprintf(stderr, "Write result cost: %.4lfs\n", write_ela);
#endif
    }
out:
#ifdef DEBUG_ENABLED
    t2 = std::chrono::high_resolution_clock::now();
    ela = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "Totoal query cost: %.4lfs\n", ela);
#endif
    return 0;
}