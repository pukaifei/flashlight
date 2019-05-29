/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/semisupervised/runtime/Defines.h"

namespace w2l {

// DATA OPTIONS
DEFINE_string(trainaudio, "", "Unpaired audio training data");
DEFINE_int64(pairediter, 1, "Number of iterations for the paired training set");
DEFINE_int64(
    audioiter,
    0,
    "Number of iterations for the unpaired audio training set");

} // namespace w2l
