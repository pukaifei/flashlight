/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/lead2Gold/src/common/Utils.h"

#include <flashlight/flashlight.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "common/Transforms.h"
#include "experimental/lead2Gold/src/common/Defines.h"

namespace w2l {

std::vector<std::string> NoisetknPrediction2Ltr(
    std::vector<int> tokens,
    const Dictionary& tokenDict) {
  if (tokens.empty()) {
    return std::vector<std::string>{};
  }

  if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kAsgCriterion ||
      FLAGS_criterion == kAsgBeamNoiseCriterion ||
      FLAGS_criterion == kCtcBeamNoiseCriterion) {
    uniq(tokens);
  }
  if (FLAGS_criterion == kCtcCriterion ||
      FLAGS_criterion == kCtcBeamNoiseCriterion) {
    int blankIdx = tokenDict.getIndex(kBlankToken);
    tokens.erase(
        std::remove(tokens.begin(), tokens.end(), blankIdx), tokens.end());
  }
  tokens = validateIdx(tokens, -1);
  remapLabels(tokens, tokenDict);

  return tknIdx2Ltr(tokens, tokenDict);
}

// void eraseTargets(std::shared_ptr<NoiseW2lListFilesDataset> ds_){
//  for (int64_t idx=0 ; idx < ds_->size() ; idx++){
//    ds_->eraseTargets(idx);
//  }
// for (auto& sample : *ds_) {
//  sample[kTargetIdx] = af::array();
// sample[kNoiseKeyIdx] = af::array();
//}
//}

} // namespace w2l
