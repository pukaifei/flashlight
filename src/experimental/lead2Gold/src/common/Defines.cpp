/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/lead2Gold/src/common/Defines.h"
#include "common/Defines.h"

#include <cstdlib>
#include <limits>

namespace w2l {
    
// ASG BEAM NOISE OPTIONS
// 
// DECLARE_double(lrnoiselm);
DEFINE_double(lrnoiselm, 0.001, "NoiseLM learning rate (ASGBeamNoise only)");
DEFINE_double(lrscalenoise, 0.0001, "Scale learning rate (ASGBeamNoise only)");
DEFINE_string(noisekeys, "", "path/to/noise/key/lexique");
DEFINE_string(noisetarget, "noisesplit", "noise target split. Separate the sentence into noisy keys");
DEFINE_string(cleantarget, "", "clean target split. Separate the sentence into clean keys");
DEFINE_bool(restartEpochIfFork, false, "set startEpoch to 0 if fork");
DEFINE_bool(allowSwap, true, "allow swap noise probability");
DEFINE_bool(allowInsertion, true, "allow insertion noise probability");
DEFINE_bool(allowDeletion, true, "allow deletion noise probability");
DEFINE_bool(storeGroundTruth, false, "if true the second tab separated string in a FileListDataset is the ground truth");
DEFINE_bool(storeForNoiselm, false, "if true kCleanNoiselmKeyIdx and kNoisyNoiselmKeyIdx fields are used");
//DEFINE_double(beamthreshold, 0, "beam score threshold. 0 means no threhold");
DEFINE_string(probasdir, "", "path/to/probas/dir");
DEFINE_string(noiselmtype, "zeronoiselm", "noise lm type");
DEFINE_bool(autoScale, false, "auto scaling of noise probabilities");
DEFINE_double(scale_noise, 0.4, "multiplicative scaling to log(p noise) if autoScale is false for substitutions");
DEFINE_double(scale_sub, 1, "additive scaling to log(p noise) if autoScale is false for substitutions");
DEFINE_double(scale_ins, 1, "additive scaling to log(p noise) if autoScale is false for insertions");
DEFINE_double(scale_del, 1, "additive scaling to log(p noise) if autoScale is false for deletions");
DEFINE_double(tkn_score, 0, "additive scaling to log(p noise) if autoScale is false for deletions");
DEFINE_bool(computeStats, false, "compute stats for each node, make the code slow");
DEFINE_int64(topk, 0, "Take top k tokens from emission scores to expand hypotheses. No pruning if topk=0");
DEFINE_bool(useevalemission, false, "compute emission with evaluation mode to get a better wLER accuracy");
DEFINE_bool(useNoiseToSort, true, "use noisescore to sort hypotheses");


DEFINE_string(saveExamplePathFolder, "", "path were to cerialize emission probas");
DEFINE_string(computeStatsStorePath, "", "path were to store results in a json format");
DEFINE_bool(computeStatsLight, false, "if true do not store the stats for every frame");
DEFINE_int64(nb_ex, 20, "Number of maximum example to process for computing stats");

DEFINE_string(statsDirection, "forward", "direction to retrieve paths");
DEFINE_int64(statsbeamsize, 10, "size of the beam when searching for the paths in stats mode");

DEFINE_int64(nbNested, 1, "Nested loop parallel mode. Nb of threads to span for each example. Make sure batchsize * nbNested <= available core ");
DEFINE_bool(identityTest, false, "if true use identity noiselm");

/* ========== OPTIMIZER OPTIONS ========== */
DEFINE_string(scaleoptim, "adam", "optimizer for the criterion");
DEFINE_int64(iterscale, 30, "number of iteration to perform to update scale");

/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */
DEFINE_int64(saveevery, 1, "if itersave is true, save model every saveevery iteration");
//DEFINE_int64(warmup, 8000, "the LR warmup parameter, in batches");
//DEFINE_bool(use_saug, false, "Use SpecAugment");
//DEFINE_int64(lr_decay, std::numeric_limits<int64_t>::max(), "Epoch for the first LR decay");
//DEFINE_int64(lr_decay_step, std::numeric_limits<int64_t>::max(), "Epochs for each new LR decay");

/* ========== DATA SCHEDULER SEMI-SUPERVISED ========== */
DEFINE_string(trainaudio, "", "Unpaired audio training data");
DEFINE_string(trainnoise, "", "Paired audio training data for the noise model");

DEFINE_int64(
    pairediter,
    0,
    "Number of steps per epoch for the paired training set. If 0 use the size of the dataset");
DEFINE_int64(
    audioiter,
    0,
    "Number of steps per epoch for the unpaired audio training set");

DEFINE_double(
    ratioaudio,
    0,
    "If ratioaudio or audioiter = 0, take maixium size. Take audioiter = ratioaudio * pairediter ");
DECLARE_double(ratioaudio);

DEFINE_int64(
    audiowarmupepochs,
    0,
    "Number of epochs to warm up the unpaired audio training set");
DEFINE_string(
    schedulerorder,
    kUniformOrder,
    "the access order between the datasets in the data scheduler (uniform, inorder, random)");
DEFINE_int64(
    audiobatchsize,
    0,
    "Batch size for the unpaired audio training data");


DEFINE_string(
    unpairedSampling,
    kModelSampling,
    "Sampling strategy to use on unpaired audio (model, gumbel)");

DEFINE_int64(
    updateTranscriptEveryNEpoch,
    0,
    "Update proposed transcription for unpaired data every n epochs");

DEFINE_bool(evalbeamnoise, false, "if true compute asgbeamnoise criterion loss on dev datasets");
DEFINE_bool(evalnoiselm, false, "if true eval noiselm");
DEFINE_bool(updateOnTheFly, false, "if true update transcriptions every time we need them");


DEFINE_int64(
    updateScaleEveryNEpoch,
    0,
    "Update beam params (scale parameter) every n epochs");

DEFINE_int64(
    updatedNoiseModelEveryNEpoch,
    0,
    "Update noise model every n epoch");

//if valid is too slow. Testing purpose
DEFINE_int64(
    evaluateValidEveryNEpoch,
    1,
    "Evaluate on the valid dataset every N epochs.");

DEFINE_bool(useSinPosEmb, false, "use sinus positional embedding instead of learning it");
DEFINE_bool(usePosEmbEveryLayer, false, "use positional embedding at every layer");
DEFINE_int64(XFromLayerN, 1, "get acoustic features from the Nth layer");
DEFINE_double(UseCopy, 0, "prc of the time to use the copy/paste task");

/* ========== RNN noiselm ========== */
DEFINE_int64(
    encoderrnnlayer,
    1,
    "Number of RNN layers for the encoder");

DEFINE_double(
    encoderdropout,
    0,
    "dropout for the encoder");

} // namespace w2l
