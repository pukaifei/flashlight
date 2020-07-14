/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Generic utilities which should not depend on ArrayFire / flashlight.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/Defines.h"
#include "libraries/common/Utils.h"
#include "libraries/common/WordUtils.h"
#include "common/Utils.h"
#include "experimental/lead2Gold/src/data/NoiseW2lListFilesDataset.h"


namespace w2l {
std::vector<std::string> NoisetknPrediction2Ltr(std::vector<int>, const Dictionary&);
//void eraseTargets(std::shared_ptr<NoiseW2lListFilesDataset> ds_);

} // namespace w2l
