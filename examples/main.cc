#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include <glass/nsg/nsg.hpp>
#include <glass/searcher.hpp>

#define K_MAX 10
#define K_MIN 3

// Data Part
int N, D;
std::vector<std::vector<float> > database;

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
    float *data = new float[N * D];
    int i = 0;
    for (auto line : database) {
        for (auto element : line) {
            data[i] = element;
            i++;
        }
    }

    std::unique_ptr<glass::Builder> index = std::unique_ptr<glass::Builder>((glass::Builder *)new glass::NSG(D, "L2", 32, 50));
    index->Build(data, N);
    glass::Graph<int> graph = index->GetGraph();
    searcher = create_searcher(graph, "L2", 0);
    searcher->SetData(data, N, D);
    searcher->SetEf(32);
    // delete data;
}

// ==========================================================================================


// ==========================================================================================

// Query Part
std::vector<int> query(const std::vector<float>& q, const int k) {
    std::vector<int> res(k);
    int *resp = new int[k];
    searcher->Search(q.data(), k, resp);

    for (int i = 0; i < k; ++i) {
        res[i] = resp[i];
    }

    return res;
}

// ==========================================================================================

int main() {
    std::cin >> N >> D;
    database.resize(N, std::vector<float>(D));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            std::cin >> database[i][j];
        }
    }

    preprocess();

    std::cout << "ok" << std::endl;
    std::cout.flush();

    int k;
    std::cin >> k;

    while (true) {
        std::vector<float> q(D);
        for (int i = 0; i < D; ++i) {
            std::cin >> q[i];
        }

        if (std::cin.fail()) break; // Check for end of input

        std::vector<int> idxs = query(q, k);

        // Output k nearest indices
        for (float idx : idxs) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        std::cout.flush();
    }
    return 0;
}