//#include "flashlight/autograd/Functions.h"
//#include "flashlight/nn/Init.h"
//#include "flashlight/nn/Utils.h"
//
#include "experimental/lead2Gold/src/criterion/l2g/TransformerBlockMultiAttend.h"

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
    const fl::Variable& mask,
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
  if (!mask.isempty()) {
    scores = scores + tileAs(mask, scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn, v);
  result = moddims(result, af::dim4(-1, headDim * nHead, bsz));
  return result;
}
} // namespace

namespace w2l {

TransformerBlockMultiAttend::TransformerBlockMultiAttend(
    int32_t modelDimIn1,
    int32_t modelDimIn2,
    int32_t modelDimOut,
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
      w1_(std::make_shared<fl::Linear>(transformerInitLinear(modelDimOut, mlpDim))),
      w2_(std::make_shared<fl::Linear>(transformerInitLinear(mlpDim, modelDimOut))),
      wq_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))), // 2nd dim is in fact headDimAuto * nHeads
      wk_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))), // 2nd dim is in fact headDimAuto * nHeads
      wv_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))), // 2nd dim is in fact headDimAuto * nHeads
      wf_auto_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))), // 2nd dim is in fact headDimAuto * nHeads
      wq1_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn1, modelDimOut))), // 2nd dim is in fact headDim1 * nHeads
      wk1_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn1, modelDimOut))), // 2nd dim is in fact headDim1 * nHeads
      wv1_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn1, modelDimOut))), // 2nd dim is in fact headDim1 * nHeads
      wf1_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))), // 1st dim is in fact headDim1 * nHeads
      wq2_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn1, modelDimOut))),
      wk2_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn2, modelDimOut))),
      wv2_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimIn2, modelDimOut))),
      wf2_(std::make_shared<fl::Linear>(
          transformerInitLinear(modelDimOut, modelDimOut))),
      norm1_(std::make_shared<fl::LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<fl::LayerNorm>(std::vector<int>({0, 3}))),
      norm3_(std::make_shared<fl::LayerNorm>(std::vector<int>({0, 3}))) {
  if (usePosEmb_){
    params_.push_back(fl::uniform(2 * bptt - 1, modelDimOut / nHeads, -0.1, 0.1));
  }

  add(w1_);
  add(w2_);
  add(wq_auto_);
  add(wk_auto_);
  add(wv_auto_);
  add(wf_auto_);
  add(wq1_);
  add(wk1_);
  add(wv1_);
  add(wf1_);
  add(wq2_);
  add(wk2_);
  add(wv2_);
  add(wf2_);
  add(norm1_);
  add(norm2_);
  add(norm3_);
}

fl::Variable TransformerBlockMultiAttend::mlp(const fl::Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(fl::dropout(fl::relu((*w1_)(input)), pDropout));
}

fl::Variable TransformerBlockMultiAttend::getMask(int32_t n, bool cache) {
  auto mask = af::lower(af::constant(1.0, n, n), true);
  if (cache) {
    auto maskCache = af::upper(af::constant(1.0, n, n));
    mask = af::join(1, maskCache, mask);
  }
  return fl::Variable(af::log(mask), false);
}

fl::Variable TransformerBlockMultiAttend::selfAttention(const std::vector<fl::Variable>& input, const int32_t offset = 0) {
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = fl::transpose((*wq_auto_)(input.back()));
  auto k = fl::transpose((*wk_auto_)(fl::concatenate(input, 1)));
  auto v = fl::transpose((*wv_auto_)(fl::concatenate(input, 1)));

  fl::Variable mask, posEmb;
  if (usePosEmb_){
    posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
  }
  if (input.back().dims(1) > 1) {
    mask = getMask(n, input.size() == 2);
  }

  auto result = transformerMultiheadAttention(
      q, k, v, posEmb, mask, nHeads_, pDrop, offset);
  result = (*wf_auto_)(fl::transpose(result));

  return result;
}


fl::Variable TransformerBlockMultiAttend::Attention(const fl::Variable& input, const fl::Variable& encoded, const int32_t offset = 0, const int32_t numAttend = 1) {
  int n = input.dims(1), bsz = input.dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  fl::Variable q,k,v;

  if ( numAttend == 1 ){
    q = fl::transpose((*wq1_)(input));
    k = fl::transpose((*wk1_)(encoded));
    v = fl::transpose((*wv1_)(encoded));
  } else if ( numAttend == 2 ){
    q = fl::transpose((*wq2_)(input));
    k = fl::transpose((*wk2_)(encoded));
    v = fl::transpose((*wv2_)(encoded));
  }

  fl::Variable mask, posEmb; // mask and posEmb unused
  //posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));

  auto result = transformerMultiheadAttention(
      q, k, v, posEmb, mask, nHeads_, pDrop, offset);

  if ( numAttend == 1 ){
    result = (*wf1_)(fl::transpose(result));
  } else if ( numAttend == 2 ){
    result = (*wf2_)(fl::transpose(result));
  }  

  return result;
}


std::vector<fl::Variable> TransformerBlockMultiAttend::forward(const std::vector<fl::Variable>& input, const std::vector<fl::Variable>& encodedMulti) {
  auto x = input.back();
  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }

  int offset = (input.size() == 1) ? 0 : input[0].dims(1);

  auto h = (*norm1_)(f * selfAttention(input, offset) + x);
  h = (*norm2_)(f * ( Attention(h, encodedMulti[0], offset, 1) + Attention(h, encodedMulti[1], offset, 2) ) + h);

  return {(*norm3_)(f * mlp(h) + h)};
}

std::vector<fl::Variable> TransformerBlockMultiAttend::forward(const std::vector<fl::Variable>& input) { 
  return forward(input, input);
}

std::string TransformerBlockMultiAttend::prettyString() const {
  return "TransformerBlockMultiAttend";
}

TransformerBlockMultiAttend::TransformerBlockMultiAttend() {}

} // namespace fl