/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>
#include <glog/logging.h>
#include <fstream>
#include <string>

#include "common/FlashlightUtils.h"

namespace w2l {

void initializeWeights(const std::string& fileName, fl::Variable& convWt) {
  LOG(INFO) << "Loading weights from " << fileName << "\n";
  std::ifstream file(fileName);

  if (!file.is_open()) {
    LOG(FATAL) << "unable to open lowpass weights file";
  }

  auto wtDims = convWt.dims();
  auto N = convWt.elements();

  std::vector<float> weightVec(N);
  std::string line;

  int i = 0;
  while (file >> line) {
    if (i == N) {
      LOG(FATAL) << "Weight file contains more entries than expected ";
    }
    try {
      weightVec[i] = std::stod(line);
    } catch (const std::out_of_range& oor) {
      LOG(ERROR) << "weight out of range, using 0.0 instead";
    }
    ++i;
  }
  convWt.array() = af::array(wtDims, weightVec.data());
  file.close();
}

} // namespace w2l
