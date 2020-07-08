/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <sstream>
#include <string>

#include "experimental/frontend/Lowpass.h"

using namespace fl;

namespace w2l {

Lowpass::Lowpass(int nin, int kw, int dw, LPFMode lpfMode)
    : Conv2D(nin, nin, kw, 1, dw, 1, 0, 0, 1, 1, false, nin),
      lpfMode_(lpfMode) {
  if (lpfMode != LPFMode::LEARN && lpfMode != LPFMode::FIXED) {
    LOG(FATAL) << "mode should be either 'FIXED' or 'LEARN'";
  }

  auto lowpassValues = sqHanning(xFilter_);
  setParams(fl::Variable(af::tile(lowpassValues, 1, 1, 1, nIn_), false), 0);
}

af::array Lowpass::sqHanning(int T) {
  auto t = range(af::dim4(T));
  auto v = 0.5 * (1 - af::cos(2 * M_PI * t / (T - 1)));
  return v * v;
}

void Lowpass::train() {
  train_ = true;
  if (lpfMode_ != LPFMode::FIXED) {
    Conv2D::train();
  }
}

std::string Lowpass::prettyString() const {
  std::ostringstream ss;
  ss << "Lowpass";
  ss << " (" << Conv2D::prettyString() << ")";
  return ss.str();
}

} // namespace w2l
