/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include <string>
#include <unordered_map>

#include "common/Dictionary.h"
#include "experimental/semisupervised/src/module/LMCritic.h"

namespace w2l {
std::unordered_map<std::string, std::string> setFlags(int argc, char** argv);

std::shared_ptr<fl::Module> initLM(const Dictionary& lmDict);

std::shared_ptr<LMCritic> createLMCritic(
    const Dictionary& lmDict,
    const Dictionary& amDict);

} // namespace w2l
