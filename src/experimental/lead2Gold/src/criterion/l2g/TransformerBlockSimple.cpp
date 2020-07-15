//#include "flashlight/autograd/Functions.h"
//#include "flashlight/nn/Init.h"
//#include "flashlight/nn/Utils.h"
//
#include "experimental/lead2Gold/src/criterion/l2g/TransformerBlockSimple.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

fl::Variable transformerRotate(const fl::Variable& input) {
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  auto gradFunc = [d0, d1, d2, d3](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    auto gradData = gradOutput.array();
    gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    gradData = af::join(0, gradData, af::constant(0.0, d1, 1, d2, d3));
    gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    gradData = gradData.rows(0, d0 - 1);
    inputs[0].addGrad(fl::Variable(gradData, false));
  };
  return fl::Variable(data, {input}, gradFunc);
}

fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const int32_t nHead,
    const double pDropout,
    const int32_t offset = 0) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHead;

  auto q = moddims(query, af::dim4(-1, headDim, nHead * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHead * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHead * bsz));

  auto scores = matmulNT(q, k);
  if (!posEmb.isempty()) {
    int n = posEmb.dims(0) / 2 - offset;
    auto pscores = transformerRotate(matmulNT(posEmb, q));
    scores = scores + transpose(pscores.rows(n, n + k.dims(0) - 1));
  }
  scores = scores / std::sqrt(float(headDim));

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn, v);
  result = moddims(result, af::dim4(-1, headDim * nHead, bsz));
  return result;
}
} // namespace

namespace w2l {

TransformerBlockSimple::TransformerBlockSimple(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    float pDropout,
    float pLayerdrop,
    bool usePosEmb)
    : nHeads_(nHeads),
      bptt_(bptt),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      usePosEmb_(usePosEmb),
      w1_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<fl::Linear>(
          transformerInitLinear(mlpDim, modelDim))),
      wq_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wk_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wv_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wf_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<fl::LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<fl::LayerNorm>(std::vector<int>({0, 3}))) {
  if (usePosEmb_) {
    params_.push_back(fl::uniform(2 * bptt - 1, modelDim / nHeads, -0.1, 0.1));
  }
  add(w1_);
  add(w2_);
  add(wq_auto_);
  add(wk_auto_);
  add(wv_auto_);
  add(wf_auto_);
  add(norm1_);
  add(norm2_);
}

fl::Variable TransformerBlockSimple::mlp(const fl::Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(fl::dropout(fl::relu((*w1_)(input)), pDropout));
}

fl::Variable TransformerBlockSimple::selfAttention(
    const std::vector<fl::Variable>& input,
    const int32_t offset = 0) {
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = fl::transpose((*wq_auto_)(input.back()));
  auto k = fl::transpose((*wk_auto_)(fl::concatenate(input, 1)));
  auto v = fl::transpose((*wv_auto_)(fl::concatenate(input, 1)));

  fl::Variable posEmb;
  if (usePosEmb_) {
    posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
  }

  auto result =
      transformerMultiheadAttention(q, k, v, posEmb, nHeads_, pDrop, offset);
  result = (*wf_auto_)(fl::transpose(result));

  return result;
}

std::vector<fl::Variable> TransformerBlockSimple::forward(
    const std::vector<fl::Variable>& input,
    const std::vector<fl::Variable>& encoded) {
  auto x = input.back();
  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }

  int offset = (input.size() == 1) ? 0 : input[0].dims(1);

  auto h = (*norm1_)(f * selfAttention(input, offset) + x);
  return {(*norm2_)(f * mlp(h) + h)};
}

std::vector<fl::Variable> TransformerBlockSimple::forward(
    const std::vector<fl::Variable>& input) {
  return forward(input, input);
}

std::string TransformerBlockSimple::prettyString() const {
  return "TransformerBlockSimple";
}

TransformerBlockSimple::TransformerBlockSimple() {}

} // namespace w2l