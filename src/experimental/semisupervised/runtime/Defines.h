/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <gflags/gflags.h>

#include "common/Defines.h"

namespace w2l {

// config
constexpr const char* kIteration = "iteration";
constexpr const char* kReloadPath = "reloadPath";
constexpr const char* kRunStatus = "runStatus";
constexpr const char* kStartEpoch = "startEpoch";
constexpr const char* kStartIter = "startIter";

// meter
constexpr const char* kTarget = "target";
constexpr const char* kASR = "asr";
constexpr const char* kLM = "lm";
constexpr const char* kFullModel = "fullModel";
constexpr const char* kRuntime = "runtime";
constexpr const char* kTimer = "bch";
constexpr const char* kSampleTimer = "smp";
constexpr const char* kFwdTimer = "fwd";
constexpr const char* kCritFwdTimer = "crit-fwd";
constexpr const char* kBwdTimer = "bwd";
constexpr const char* kOptimTimer = "optim";

// data
// continue from src/common/Defines.h
constexpr size_t kDataTypeIdx = kNumDataIdx;
constexpr size_t kGlobalBatchIdx = kNumDataIdx + 1;
constexpr size_t kParallelData = 1;
constexpr size_t kUnpairedAudio = 2;

// flags
DECLARE_string(trainaudio);
DECLARE_int64(pairediter);
DECLARE_int64(audioiter);

} // namespace w2l
