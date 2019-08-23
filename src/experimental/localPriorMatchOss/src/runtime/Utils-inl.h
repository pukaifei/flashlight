/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace w2l {

template <class T>
std::string stringify(const std::vector<T>& vec, std::string sep) {
  std::ostringstream os;
  for (auto& val : vec) {
    os << val << sep;
  }
  return os.str();
}

template<class T, class S>
std::vector<S> getLengths(const std::vector<std::vector<T>>& vec) {
  std::vector<S> lengths;
  if (vec.empty()) {
    lengths.push_back(0);
  } else {
    for (auto& val : vec) {
      lengths.push_back(val.size());
    }
  }
  return lengths;
}

} // namespace w2l
