/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <unordered_map>

#include "data/Featurize.h"
#include "data/Sound.h"
#include "libraries/common/Dictionary.h"
#include "libraries/feature/FeatureParams.h"

namespace w2l {

W2lFeatureData featurize2(
    const std::vector<W2lLoaderData>& data,
    const DictionaryMap& dicts);

} // namespace w2l
