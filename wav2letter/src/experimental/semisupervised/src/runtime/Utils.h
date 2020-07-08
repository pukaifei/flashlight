/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <utility>

#include "libraries/common/Dictionary.h"

namespace w2l {
Dictionary createFairseqTokenDict(const std::string& filepath);

/**
 * Generate mapping between the indices of tokens in dict1 and dict2
 * for matching the dictionaries in w2l and fairseq.
 * The function returns (mapping, numPadding), where the
 * token with index i in dict1 maps to the token with index mapping[i] in dict2.
 * numPadding is the number of tokens that appear in dict1 but not dict2,
 * and we map those tokens to dict2.indexSize() + 0, dict2.indexSize() + 1, ...
 * dict2.indexSize() + numPadding - 1 in order.
 */
std::pair<std::vector<int>, int> genTokenDictIndexMap(
    const Dictionary& dict1,
    const Dictionary& dict2);

} // namespace w2l
