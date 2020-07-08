/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/Sound.h"
#include "experimental/augmentation/AudioAugmenter.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

TEST(AudioAugmenterTest, SpeedAug) {
  std::string filename = "test.wav";
  std::string fullpath = loadPath + "/" + filename;
  auto data = loadSound<float>(fullpath);
  auto info = loadSoundInfo(fullpath);
  ASSERT_EQ(data.size(), info.frames * info.channels);

  std::vector<double> speeds = {0.8, 1.2};
  for (const auto& speed : speeds) {
    auto output = speedAug(data, speed, info.channels);
    ASSERT_NEAR(output.size(), data.size() / speed, 10);
    saveSound(
        "/tmp/" + std::to_string(speed) + "x_" + filename,
        output,
        info.samplerate,
        info.channels,
        SoundFormat::WAV,
        SoundSubFormat::PCM_16);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
