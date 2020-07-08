#pragma once

#include "flashlight/nn/modules/Container.h"
#include "flashlight/nn/modules/LayerNorm.h"
#include "flashlight/nn/modules/Linear.h"
#include "flashlight/nn/modules/Module.h"

namespace fl {

class TransformerLayer : public Container {
 public:
  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t nHeads_;
  int32_t bptt_;
  double pDropout_;
  double pLayerdrop_;
  bool useMask_;
  bool preLN_;
  std::shared_ptr<Linear> w1_, w2_, wq_, wk_, wv_, wf_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w1_,
      w2_,
      wq_,
      wk_,
      wv_,
      wf_,
      norm1_,
      norm2_,
      nHeads_,
      pDropout_,
      pLayerdrop_,
      bptt_,
      useMask_)

  TransformerLayer();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::TransformerLayer);
