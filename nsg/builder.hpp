#pragma once

#include "nsg/graph.hpp"

namespace nsg {

struct Builder {
  virtual void Build(float *data, int nb) = 0;
  virtual Graph<int> GetGraph() = 0;
  virtual ~Builder() = default;
};

} // namespace nsg