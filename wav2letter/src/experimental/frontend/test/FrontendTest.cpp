/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <experimental/frontend/Frontend.h>
#include <flashlight/flashlight.h>
#include <flashlight/nn/Utils.h>

#include "module/module.h"
#include "runtime/runtime.h"

using namespace fl;
using namespace w2l;

namespace {

std::string getTmpPath(const std::string& key) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  return std::string("/tmp/test_") + userstr + key + std::string(".mdl");
}
} // namespace

TEST(FrontendTest, Lowpass) {
  std::array<float, 24> in = {0.5219, 0.6762, 0.6994, 0.5294, 0.2699, 0.1551,
                              0.6745, 0.7979, 0.9081, 0.8825, 0.6852, 0.7410,
                              0.2117, 0.9544, 0.3173, 0.6407, 0.2673, 0.8299,
                              0.1205, 0.3047, 0.9969, 0.7627, 0.0804, 0.3485};
  // w h c b
  auto input = Variable(af::array(6, 1, 4, 1, in.data()), false);
  LPFMode learn = LPFMode::LEARN;

  auto net1 = Sequential();
  // nin, kw, dw
  auto lp_module = Lowpass(4, 4, 1, learn);
  net1.add(lp_module);
  auto output = net1.forward(input);
  // Outputs from torch function
  std::array<float, 12> out = {0.7738,
                               0.6912,
                               0.4496,
                               0.9596,
                               1.0072,
                               0.8818,
                               0.7153,
                               0.5389,
                               0.5107,
                               0.7321,
                               0.9898,
                               0.4743};
  auto out_data = af::array(3, 1, 4, 1, out.data());
  ASSERT_TRUE(allClose(output.array(), out_data, 1E-2));
}

TEST(SerializationTest, Lowpass) {
  std::array<float, 24> in = {0.5219, 0.6762, 0.6994, 0.5294, 0.2699, 0.1551,
                              0.6745, 0.7979, 0.9081, 0.8825, 0.6852, 0.7410,
                              0.2117, 0.9544, 0.3173, 0.6407, 0.2673, 0.8299,
                              0.1205, 0.3047, 0.9969, 0.7627, 0.0804, 0.3485};
  // w h c b
  auto input = Variable(af::array(6, 1, 4, 1, in.data()), false);
  LPFMode learn = LPFMode::FIXED;
  auto lp_module = std::make_shared<Lowpass>(4, 3, 1, learn);

  save(getTmpPath("Lowpass"), lp_module);

  std::shared_ptr<Lowpass> lp_module1;
  load(getTmpPath("Lowpass"), lp_module1);
  ASSERT_TRUE(lp_module1);

  ASSERT_TRUE(allParamsClose(*lp_module1, *lp_module));
  ASSERT_TRUE(allClose(lp_module1->forward(input), lp_module->forward(input)));
}

TEST(FrontendTest, SqL2Pooling) {
  std::array<float, 24> in = {5, 5, 2, 2};
  // w h c b
  auto input = Variable(af::array(1, 1, 2, 2, in.data()), false);
  auto net = Sequential();
  // nin, kw, dw
  auto fl2p = SqL2Pooling();
  net.add(fl2p);
  auto output = net.forward(input);
  std::array<float, 2> out = {50.0, 8.0};
  auto out_data = af::array(1, 1, 1, 2, out.data());
  ASSERT_TRUE(allClose(output.array(), out_data));
}

TEST(FrontendTest, LogCompression) {
  std::array<float, 24> in = {5, -5, 2, -2};
  // w h c b
  auto input = Variable(af::array(2, 1, 1, 2, in.data()), false);
  auto net = Sequential();
  // nin, kw, dw
  auto logcomp = LogCompression(0.1);
  net.add(logcomp);

  auto output = net.forward(input);
  std::array<float, 4> out = {1.6292, 1.6292, 0.7419, 0.7419};
  auto out_data = af::array(2, 1, 1, 2, out.data());
  ASSERT_TRUE(allClose(output.array(), out_data, 1E-2));
}

TEST(FrontendTest, TrainableFrontendEnd2End) {
  std::array<float, 48> in;
  in.fill(1.0);

  int feats = 8;
  int w = 6;
  int lp_kw = 4;

  auto input = Variable(af::array(w, 1, feats, 1, in.data()), true);
  double a = 0.1;
  for (int i = 0; i < feats; i++) {
    for (int j = 0; j < w; j++) {
      input.array()(j, 0, i, 0) += a;
      a += 0.1;
    }
  }

  auto net = Sequential();
  net.add(SqL2Pooling());
  LPFMode learn = LPFMode::FIXED;
  auto lpf = Lowpass(feats / 2, lp_kw, 1, learn);
  net.add(lpf);
  net.add(LogCompression(1.0));

  auto output = net.forward(input);
  output.backward();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
