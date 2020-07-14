/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include "experimental/lead2Gold/src/common/Defines.h"

namespace w2l {

std::shared_ptr<fl::FirstOrderOptimizer> initParamOptimizer(
    const std::vector<fl::Variable> params,
    const std::string& optimizer,
    double lr,
    double momentum,
    double weightdecay);
} // namespace w2l
