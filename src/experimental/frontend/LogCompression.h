/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

class LogCompression : public fl::UnaryModule {
 private:
  LogCompression() {}
  double m_k_;

 public:
  explicit LogCompression(double k) : m_k_(k) {}

  fl::Variable forward(const fl::Variable& input) override;

  FL_SAVE_LOAD_WITH_BASE(fl::UnaryModule, m_k_)

  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::LogCompression)
