#include "experimental/compat/PositionEmbeddingLayer.h"

namespace fl {

std::vector<Variable> PositionEmbeddingLayer::forward(
    const std::vector<Variable>& /* input */) {
  throw std::runtime_error(
      "PositionEmbeddingLayer is deprecated - use PositionEmbedding");
  return std::vector<Variable>();
}

std::string PositionEmbeddingLayer::prettyString() const {
  return "PositionEmbeddingLayer";
}

PositionEmbeddingLayer::PositionEmbeddingLayer() {}

} // namespace fl
