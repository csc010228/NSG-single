#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "nsg/graph.hpp"
#include "nsg/neighbor.hpp"
#include "nsg/simd/distance.hpp"
#include "nsg/utils.hpp"

namespace nsg {

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
    std::vector<int> eval_points(num_eval);
    std::vector<std::vector<int>> eval_gt(num_eval);
    std::mt19937 rng(random_seed * 6577);
    GenRandom(rng, eval_points.data(), num_eval, nb);
    GenEvalGt(eval_points, eval_gt);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 1; iter <= iters; ++iter) {
      Join();
      Update();
      float recall = EvalRecall(eval_points, eval_gt);
      printf("NNDescent iter: [%d/%d], recall: %f\n", iter, iters, recall);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ela = std::chrono::duration<double>(t2 - t1).count();
    printf("NNDescent cost: %.2lfs\n", ela);
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
      // TO BE NOTED
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

} // namespace nsg