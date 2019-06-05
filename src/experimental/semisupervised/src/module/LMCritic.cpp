/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/semisupervised/src/module/LMCritic.h"

using namespace fl;

namespace w2l {

LMCritic::LMCritic(
    std::shared_ptr<Module> network,
    const std::vector<int>& dictIndexMap,
    int numDictPadding,
    int startIndex,
    int unkIndex /* = -1 */)
    : dictIndexMap_(dictIndexMap.size(), dictIndexMap.data()),
      numDictPadding_(numDictPadding),
      unkIndex_(unkIndex) {
  add(network);

  af::array startProb = af::constant(0.0, dictIndexMap.size(), f32);
  startProb(startIndex) = 1.0;
  startProb_ = Variable(startProb, false);
}

std::vector<Variable> LMCritic::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("Invalid inputs size");
  }

  // [nClass, targetlen, batchsize], expects log prob as input
  auto probInput = exp(inputs[0]);
  int U = probInput.dims(1);
  int B = probInput.dims(2);
  if (U == 0) {
    throw std::invalid_argument("Invalid input variable size");
  }

  probInput = preprocessInput(probInput);

  // pad start token
  Variable lmInput = tile(startProb_, {1, 1, B});
  if (U > 1) {
    lmInput = concatenate(
        {lmInput, probInput(af::span, af::seq(0, U - 2), af::span)}, 1);
  }

  auto logProbOutput = lmNetwork()->forward({lmInput}).front();

  // cross entropy loss
  auto losses = negate(flat(sum(probInput * logProbOutput, {0, 1})));
  return {losses, logProbOutput};
}

Variable LMCritic::preprocessInput(Variable input) {
  Variable result = input;
  // padding
  if (numDictPadding_ > 0) {
    Variable pad = constant(
        0.0,
        af::dim4({numDictPadding_, input.dims(1), input.dims(2)}),
        f32,
        false);
    result = concatenate({result, pad}, 0);
  }

  // mapping
  result = result(dictIndexMap_, af::span, af::span);

  // <unk>
  if (unkIndex_ >= 0) {
    auto unkProb =
        1.0 - sum(result, {0}) + result(unkIndex_, af::span, af::span);
    if (unkIndex_ == 0) {
      result = concatenate(
          {unkProb,
           result(
               af::range(af::dim4(result.dims(0) - 1)) + 1,
               af::span,
               af::span)},
          0);
    } else if (unkIndex_ == result.dims(0) - 1) {
      result = concatenate(
          {result(af::range(af::dim4(result.dims(0) - 1)), af::span, af::span),
           unkProb},
          0);
    } else {
      result = concatenate(
          {
              result(af::range(af::dim4(unkIndex_)), af::span, af::span),
              unkProb,
              result(
                  af::range(af::dim4(result.dims(0) - unkIndex_ - 1)) +
                      unkIndex_ + 1,
                  af::span,
                  af::span),
          },
          0);
    }
    result(unkIndex_, af::span, af::span) = unkProb;
  } else {
    result = result / tile(sum(result, {0}), {result.dims(0), 1, 1});
  }

  return result;
}

std::string LMCritic::prettyString() const {
  return "LMCritic with LM: " + lmNetwork()->prettyString();
}

} // namespace w2l
