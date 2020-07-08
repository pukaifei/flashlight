/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/frontend/LogCompression.h"

#include <cmath>
#include <sstream>
#include <string>

namespace w2l {

fl::Variable LogCompression::forward(const fl::Variable& input) {
  return fl::log(m_k_ + fl::abs(input));
}

std::string LogCompression::prettyString() const {
  std::ostringstream ss;
  ss << "LogCompression log( " << m_k_ << " + abs(input) )";
  return ss.str();
}

} // namespace w2l
