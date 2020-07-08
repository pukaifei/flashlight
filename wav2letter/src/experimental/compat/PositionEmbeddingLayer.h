#pragma once

#include "flashlight/nn/modules/Container.h"
#include "flashlight/nn/modules/LayerNorm.h"
#include "flashlight/nn/modules/Linear.h"
#include "flashlight/nn/modules/Module.h"

namespace fl {

class PositionEmbeddingLayer : public Container {
 public:
  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container, dropout_)

  double dropout_;

  friend class cereal::access;
  PositionEmbeddingLayer();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PositionEmbeddingLayer);
