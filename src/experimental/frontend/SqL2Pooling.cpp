/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>
#include <string>

#include "experimental/frontend/SqL2Pooling.h"

namespace w2l {

fl::Variable SqL2Pooling::forward(const fl::Variable& input) {
  auto ro = reorder(input, 0, 2, 1, 3);
  auto sq = ro * ro;
  auto pool_out =
      2.0 * pool2d(sq, 1, 2, 1, 2, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING);

  return reorder(pool_out, 0, 2, 1, 3);
}

std::string SqL2Pooling::prettyString() const {
  return "SqL2Pooling";
}

} // namespace w2l
