/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/semisupervised/src/runtime/Defines.h"

#include "criterion/Defines.h"

namespace w2l {

// data scheduler
DEFINE_string(trainaudio, "", "Unpaired audio training data");
DEFINE_int64(
    pairediter,
    1,
    "Number of steps per epoch for the paired training set");
DEFINE_int64(
    audioiter,
    0,
    "Number of steps per epoch for the unpaired audio training set");
DEFINE_int64(
    audiowarmupepochs,
    0,
    "Number of epochs to warm up the unpaired audio training set");
DEFINE_string(
    schedulerorder,
    kUniformOrder,
    "the access order between the datasets in the data scheduler (uniform, inorder, random)");

// lmcrit
DEFINE_string(lmdict, "", "Dictionary used in LM training");
DEFINE_string(
    lmcrit,
    kLMCECrit,
    "LM criterion type (crossEntropy, adaptiveSoftmax)");
DEFINE_string(lmarchfile, "", "LM architecture");
DEFINE_string(lmweightfile, "", "LM weights parsed from fairseq");
DEFINE_string(
    lmadasoftmaxcutoff,
    "",
    "cutoff thresholds for LM adaptiveSoftmax criterion");
DEFINE_int64(lmadasoftmaxinputdim, 0, "output dimension of LM adaptiveSoftmax");
DEFINE_int64(
    lmtempstepsize,
    1000000,
    "We multiply gumbel temperature by gamma every stepsize epochs");
DEFINE_string(
    unpairedSampling,
    kModelSampling,
    "Sampling strategy to use on unpaired audio (model, gumbel)");

} // namespace w2l
