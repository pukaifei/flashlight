/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>
#include "common/Utils.h"

#include "criterion/CriterionUtils.h"
namespace w2l {

class ForceAlignment {
 public:
  explicit ForceAlignment(const std::vector<float>& transition);

  // Code mostly copied from ForceAlignmentCriterion
  // Forwards the input through network,
  // runs viterbi using max score at each node
  // uses backptr table to compute the alignment
  std::vector<std::vector<int>> align(
      const fl::Variable& input,
      const fl::Variable& target);

 private:
  ForceAlignment() = default;
  std::vector<float> transraw_;

  struct FalParameters {
    std::vector<int> targetraw;
    std::vector<float> inputraw, transraw, scale;
    std::vector<float> fal;
    std::vector<double> falacc;
    std::vector<double> s1i, s2i;
    std::vector<int> backptr;

    FalParameters(int n, int t, int b, int l) {
      targetraw.resize(l * b);
      inputraw.resize(b * t * n);
      fal.resize(b);
      scale.resize(b);
      falacc.resize(b * l * t);
      backptr.resize(b * l * t);
      s1i.resize(b * l);
      s2i.resize(b * l);
      transraw.resize(n * n);
    }
  };
};

// typedef ForceAlignmentCriterion FACLoss;
} // namespace w2l
