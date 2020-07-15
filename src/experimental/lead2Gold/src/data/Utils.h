/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <glog/logging.h>
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/CriterionUtils.h"
#include "criterion/SequenceCriterion.h"
#include "data/Utils.h"
#include "experimental/lead2Gold/src/common/Defines.h"

namespace w2l {

// Helper class used to store data in W2lListFilesDataset
class SpeechSample2 {
 private:
  std::string sampleId_; // utterance id
  std::string audioFilePath_; // full path to audio file
  std::vector<std::string> transcript_; // word transcripts
  std::vector<std::string>
      groundTruthTranscript_; // word ground truth transcript

 public:
  SpeechSample2() {}

  SpeechSample2(
      std::string sampleId,
      std::string audioFile,
      std::vector<std::string> trans,
      std::vector<std::string> ground_truth_trans = {})
      : sampleId_(sampleId),
        audioFilePath_(audioFile),
        transcript_(std::move(trans)),
        groundTruthTranscript_(std::move(ground_truth_trans)) {}

  std::string getSampleId() const {
    return sampleId_;
  }

  std::string getAudioFile() const {
    return audioFilePath_;
  }

  int64_t numWords() const {
    return transcript_.size();
  }

  std::vector<std::string> getTranscript() const {
    return transcript_;
  }

  void setTranscript(std::vector<std::string> transcript) {
    transcript_ = transcript;
  }

  std::vector<std::string> getGroundTruthTranscript() const {
    return groundTruthTranscript_;
  }

  void setGroundTruthTranscript(
      std::vector<std::string> groundTruthTranscript) {
    groundTruthTranscript_ = groundTruthTranscript;
  }

  std::string getTranscript(int64_t id) const {
    if (id >= transcript_.size()) {
      throw std::out_of_range("getTranscript idx out of range");
    }
    return transcript_[id];
  }
};

std::vector<af::array> getUpdateTrancripts(
    fl::Variable& emissions,
    std::shared_ptr<SequenceCriterion> criterion,
    DictionaryMap& dicts,
    int padidx = -1,
    bool addeos = false);
void updateTrancripts(
    std::vector<af::array>& sample,
    fl::Variable& emissions,
    std::shared_ptr<SequenceCriterion> criterion,
    DictionaryMap& dicts);

std::vector<std::vector<std::string>> getUpdateTrancriptsWords(
    fl::Variable& emissions,
    std::shared_ptr<SequenceCriterion> criterion,
    DictionaryMap& dicts);

} // namespace w2l
