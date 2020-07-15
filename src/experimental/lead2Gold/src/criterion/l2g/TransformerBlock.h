#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

/**
 * A module which implements a Transformer Block.
 *
 * For details, see [Vaswani et al
 * (2017)](https://arxiv.org/abs/1706.03762).
 *
 * This module also supports layer drop regularization, as introduced in
 * [Fan et al (2019)](https://arxiv.org/abs/1909.11556).
 *
 * Input dimension at forward is assumed to be CxUxBx1, where C is the
 * number of features, U the sequence length and B the batch size.
 *
 */

class TransformerBlock : public fl::Container {
 public:
  TransformerBlock(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t bptt,
      float pDropout,
      float pLayerdrop,
      bool usePosEmb);

  // if input is of size 2. First element is the past and we ask to forward the
  // 2nd element. if input is of size 1. We compute every possibility. encoded is
  // a sequence CxTxBx1 on which to attend.
  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input,
      const std::vector<fl::Variable>& encoded);
  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t nHeads_;
  int32_t bptt_;
  double pDropout_;
  double pLayerdrop_;
  bool usePosEmb_;
  std::shared_ptr<fl::Linear> w1_, w2_, wq_auto_, wk_auto_, wv_auto_, wf_auto_,
      wq2_, wk2_, wv2_, wf2_;
  // std::shared_ptr<fl::Linear> w1_, w2_, wq_auto_, wk_auto_, wv_auto_,
  // wf_auto_; std::vector<std::shared_ptr<fl::Linear>> wqs_, wks_, wvs_, wfs_;
  std::shared_ptr<fl::LayerNorm> norm1_, norm2_, norm3_;

  fl::Variable mlp(const fl::Variable& input);
  fl::Variable getMask(int32_t n, bool cache = false);
  fl::Variable selfAttention(
      const std::vector<fl::Variable>& input,
      const int32_t offset);
  fl::Variable Attention(
      const fl::Variable& input,
      const std::vector<fl::Variable>& encoded,
      const int32_t offset);

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w1_,
      w2_,
      wq_auto_,
      wk_auto_,
      wv_auto_,
      wf_auto_,
      wq2_,
      wk2_,
      wv2_,
      wf2_,
      norm1_,
      norm2_,
      norm3_,
      nHeads_,
      pDropout_,
      pLayerdrop_,
      bptt_,
      usePosEmb_)

  TransformerBlock();
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::TransformerBlock);