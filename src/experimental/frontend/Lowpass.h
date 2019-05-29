/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <flashlight/flashlight.h>
#include <flashlight/nn/modules/Conv2D.h>
#include <glog/logging.h>

namespace w2l {

enum class LPFMode {
  /// FIXED mode for not learning the low pass filter weights
  FIXED = 0,

  /// LEARN mode will learn the low pass filter weights
  LEARN = 1,
};

class Lowpass : public fl::Conv2D {
 private:
  Lowpass() {} // intentionally private

  friend class cereal::access;
  LPFMode lpfMode_;
  fl::Variable wt_;

  af::array sqHanning(int T);

 public:
  Lowpass(int nin, int kw, int dw, LPFMode lpfMode);

  void train() override;

  FL_SAVE_LOAD_WITH_BASE(fl::Conv2D, lpfMode_, wt_)

  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::Lowpass)
