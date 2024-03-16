#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "nsg/memory.hpp"
#include "nsg/simd/distance.hpp"

namespace nsg {

constexpr int EMPTY_ID = -1;

// K-Nearest Neighbors Graph
template <typename node_t> struct Graph {
  int N;                // 节点个数
  int K;                // 最近邻节点个数

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

  // 获取编号u的数据的所有出边
  const int *edges(int u) const { return data + K * u; }
  int *edges(int u) { return data + K * u; }

  // 获取边<i, j>
  node_t at(int i, int j) const { return data[i * K + j]; }
  node_t &at(int i, int j) { return data[i * K + j]; }

  void prefetch(int u, int lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  template <typename Pool, typename Computer>
  void initialize_search(Pool &pool, const Computer &computer) const {
    for (auto ep : eps) {
      pool.insert(ep, computer(ep));
    }
  }

  void save(const std::string &filename) const {
    // static_assert(std::is_same_v<node_t, int32_t>);
    static_assert(std::is_same<node_t, int32_t>::value);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, N * K * 4);
    printf("Graph Saving done\n");
  }

  void load(const std::string &filename) {
    // static_assert(std::is_same_v<node_t, int32_t>);
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
    printf("Graph Loding done\n");
  }
};

} // namespace nsg