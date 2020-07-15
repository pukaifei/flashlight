/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gflags/gflags.h>

#include "common/Defines.h"
#include "criterion/Defines.h"

#define W2L_VERSION "0.1"

namespace w2l {

// Dataset indices
// If a new field is added, `kNumDataIdx` should be modified accordingly.

constexpr size_t kNoiseKeyIdx =
    4; // Same as kTargetIdx but with replabels removed
constexpr size_t kCleanKeyIdx = 5;
constexpr size_t kNoisyNoiselmKeyIdx = 6;
constexpr size_t kCleanNoiselmKeyIdx = 7;
// constexpr size_t kNumDataIdx = 6; // total number of dataset indices

// Various constants used in w2l
constexpr const char* kAsgBeamNoiseCriterion = "asgbeamnoise";
constexpr const char* kCtcBeamNoiseCriterion = "ctcbeamnoise";
constexpr const char* kAsgBeamNoiseAnalysis = "asgbeamnoise_analysis";

// constexpr int kMaxDevicePerNode = 2;
//
// config
constexpr const char* kIteration = "iteration";
constexpr const char* kReloadPath = "reloadPath";
constexpr const char* kRunStatus = "runStatus";
constexpr const char* kStartEpoch = "startEpoch";
constexpr const char* kStartIter = "startIter";
// constexpr const char* kForkAMMode = "forkam";

// meter
constexpr const char* kTarget = "L";
constexpr const char* kWord = "W";
constexpr const char* kASRPaired = "p";
constexpr const char* kASRUnpaired = "unp";
constexpr const char* klossScale = "train-loss-sc";
constexpr const char* klossNoiselm = "dev-clean-noiselm";
constexpr const char* kRuntime = "runtime";
constexpr const char* kTimer = "bch";
constexpr const char* kSampleTimer = "smp";
constexpr const char* kFwdTimer = "fwd";
constexpr const char* kCritFwdTimer = "crit-fwd";
constexpr const char* kCritFwdNoiseTimer = "crit-noise-fwd";
constexpr const char* kUpdateTransTimer = "upd-trans";
constexpr const char* kUpdateScaleTimer = "upd-sc";
constexpr const char* kUpdateNoiseModelTimer = "upd-noiselm";
// constexpr const char* kUpdates = "updates";

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

/* ========== ASG BEAM NOISE OPTIONS ========== */

DECLARE_double(lrnoiselm);
DECLARE_double(lrscalenoise);
DECLARE_string(noisekeys);
DECLARE_string(noisetarget);
DECLARE_string(cleantarget);
DECLARE_bool(restartEpochIfFork);
DECLARE_bool(allowSwap);
DECLARE_bool(allowInsertion);
DECLARE_bool(allowDeletion);
DECLARE_double(beamthreshold);
DECLARE_string(noiselmtype);
DECLARE_string(probasdir);
DECLARE_bool(autoScale);
DECLARE_double(scale_noise);
DECLARE_double(scale_sub);
DECLARE_double(scale_ins);
DECLARE_double(scale_del);
DECLARE_double(tkn_score);
DECLARE_bool(storeGroundTruth);
DECLARE_bool(storeForNoiselm);
DECLARE_bool(computeStats);
DECLARE_int64(topk);
DECLARE_bool(useevalemission);
DECLARE_bool(useNoiseToSort);

DECLARE_string(statsDirection);
DECLARE_int64(statsbeamsize);

DECLARE_string(saveExamplePathFolder);
DECLARE_string(computeStatsStorePath);
DECLARE_bool(computeStatsLight);
DECLARE_int64(nb_ex);

DECLARE_int64(nbNested);
DECLARE_bool(identityTest);

/* ========== OPTIMIZER OPTIONS ========== */
DECLARE_string(scaleoptim);
DECLARE_int64(iterscale);

/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */
DECLARE_int64(saveevery);
// DECLARE_int64(warmup);
// DECLARE_bool(use_saug);
// DECLARE_int64(lr_decay);
// DECLARE_int64(lr_decay_step);

/* ========== DATA SCHEDULER SEMI-SUPERVISED ========== */
DECLARE_string(trainaudio);
DECLARE_string(trainnoise);
DECLARE_int64(pairediter);
DECLARE_int64(audioiter);
DECLARE_double(ratioaudio);
DECLARE_int64(audiowarmupepochs);
DECLARE_string(schedulerorder);
DECLARE_int64(audiobatchsize);

DECLARE_string(unpairedSampling);
DECLARE_int64(updateTranscriptEveryNEpoch);
DECLARE_bool(evalbeamnoise);
DECLARE_bool(evalnoiselm);
DECLARE_bool(updateOnTheFly);
DECLARE_int64(updateScaleEveryNEpoch);
DECLARE_int64(updatedNoiseModelEveryNEpoch);
DECLARE_int64(evaluateValidEveryNEpoch);
DECLARE_bool(useSinPosEmb);
DECLARE_bool(usePosEmbEveryLayer);
DECLARE_int64(XFromLayerN);
DECLARE_double(UseCopy);

/* ========== RNN noiselm ========== */
DECLARE_int64(encoderrnnlayer);
DECLARE_double(encoderdropout);

} // namespace w2l
