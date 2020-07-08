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
constexpr const char* kPropPath = "propPath";
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
constexpr const char* kBeamTimer = "beam";
constexpr const char* kBeamFwdTimer = "beam-fwd";
constexpr const char* kLMCritFwdTimer = "lmcrit-fwd";
constexpr const char* kBwdTimer = "bwd";
constexpr const char* kOptimTimer = "optim";
constexpr const char* kNumHypos = "num-hypo";
constexpr const char* kLMEnt = "lm-ent";
constexpr const char* kLMScore = "lm-score";
constexpr const char* kS2SEnt = "s2sent";
constexpr const char* kLen = "len";

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

// prior-matching type
constexpr const char* kRegKL = "regKl";
constexpr const char* kRevKL = "revKl";
constexpr const char* kOracle = "oracle";

// proposal-update type
constexpr const char* kNever = "never";
constexpr const char* kAlways = "always";
constexpr const char* kBetter = "better";

// lmcrit normalizing methods
constexpr const char* kNoNorm = "none";
constexpr const char* kLenNorm = "len";
constexpr const char* kUnitNorm = "unit";

// flags
// data scheduler
DECLARE_string(trainaudio);
DECLARE_int64(pairediter);
DECLARE_int64(audioiter);
DECLARE_int64(audiowarmupepochs);
DECLARE_string(schedulerorder);
DECLARE_int64(unpairedBatchsize);

// lmcrit
DECLARE_string(lmdict);
DECLARE_string(lmcrit);
DECLARE_string(lmarchfile);
DECLARE_string(lmweightfile);
DECLARE_string(lmadasoftmaxcutoff);
DECLARE_int64(lmadasoftmaxinputdim);
DECLARE_int64(lmtempstepsize);
DECLARE_string(unpairedSampling);

// within-beam prior matching
DECLARE_int64(pmBeamsz);
DECLARE_string(pmType);
DECLARE_bool(pmLabelSmooth);
DECLARE_double(hyplenratiolb);
DECLARE_double(hyplenratioub);
DECLARE_double(lmcritsmooth);
DECLARE_bool(useuniformlm);
DECLARE_double(advmargin);
DECLARE_string(normlmcritprob);
DECLARE_string(norms2sprob);
DECLARE_string(proppath);
DECLARE_string(propupdate);

// misc, for debugging
DECLARE_bool(debug);

// for ablation studies
DECLARE_bool(shuflmprob);
} // namespace w2l
