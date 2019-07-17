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
constexpr const char* kForkAMMode = "forkam";

// meter
constexpr const char* kTarget = "L";
constexpr const char* kWord = "W";
constexpr const char* kASR = "asr";
constexpr const char* kLM = "lm";
constexpr const char* kFullModel = "fullModel";
constexpr const char* kRuntime = "runtime";
constexpr const char* kTimer = "bch";
constexpr const char* kSampleTimer = "smp";
constexpr const char* kFwdTimer = "fwd";
constexpr const char* kCritFwdTimer = "crit-fwd";
constexpr const char* kLMCritFwdTimer = "lmcrit-fwd";
constexpr const char* kBwdTimer = "bwd";
constexpr const char* kOptimTimer = "optim";

// data
// continue from src/common/Defines.h
constexpr size_t kDataTypeIdx = kNumDataIdx;
constexpr size_t kGlobalBatchIdx = kNumDataIdx + 1;
constexpr size_t kParallelData = 1;
constexpr size_t kUnpairedAudio = 2;
constexpr const char* kRandomOrder = "random";
constexpr const char* kInOrder = "inorder";
constexpr const char* kUniformOrder = "uniform";

// language model
constexpr const char* kLMCECrit = "crossEntropy";
constexpr const char* kLMASCrit = "adaptiveSoftmax";

// flags
// data scheduler
DECLARE_string(trainaudio);
DECLARE_int64(pairediter);
DECLARE_int64(audioiter);
DECLARE_int64(audiowarmupepochs);
DECLARE_string(schedulerorder);
DECLARE_int64(audiobatchsize);

// lmcrit
DECLARE_string(lmdict);
DECLARE_string(lmcrit);
DECLARE_string(lmarchfile);
DECLARE_string(lmweightfile);
DECLARE_string(lmadasoftmaxcutoff);
DECLARE_int64(lmadasoftmaxinputdim);
DECLARE_int64(lmtempstepsize);
DECLARE_string(unpairedSampling);
DECLARE_bool(lmmaskpadding);

} // namespace w2l
