#include "experimental/compat/TransformerLayer.h"

namespace fl {

std::vector<Variable> TransformerLayer::forward(
    const std::vector<Variable>& /* input */) {
  throw std::runtime_error("TransformerLayer is deprecated - use Transformer");
  return std::vector<Variable>();
}

std::string TransformerLayer::prettyString() const {
  return "TransformerLayer";
}

TransformerLayer::TransformerLayer() {}

} // namespace fl
