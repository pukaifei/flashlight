/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <string.h>
#include <functional>
#include <numeric>

#include "experimental/lead2Gold/src/common/Defines.h"
#include "experimental/lead2Gold/src/data/NoiseW2lListFilesDataset.h"

namespace w2l {

NoiseW2lListFilesDataset::NoiseW2lListFilesDataset(
    const std::string& filenames,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon,
    int64_t batchSize /* = 1 */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */,
    const std::string& rootdir /* = "" */)
    : W2lDataset(dicts, batchSize, worldRank, worldSize),
      lexicon_(lexicon),
      fallback2Ltr_(fallback2Ltr),
      skipUnk_(skipUnk) {
  includeWrd_ = (dicts.find(kWordIdx) != dicts.end());

  LOG_IF(FATAL, dicts.find(kTargetIdx) == dicts.end())
      << "Target dictionary does not exist";

  auto filesVec = split(',', filenames);
  std::vector<SpeechSampleMetaInfo> speechSamplesMetaInfo;
  for (const auto& f : filesVec) {
    auto fullpath = pathsConcat(rootdir, trim(f));
    auto fileSampleInfo = loadListFile(fullpath);
    speechSamplesMetaInfo.insert(
        speechSamplesMetaInfo.end(),
        fileSampleInfo.begin(),
        fileSampleInfo.end());
  }

  filterSamples(
      speechSamplesMetaInfo,
      FLAGS_minisz,
      FLAGS_maxisz,
      FLAGS_mintsz,
      FLAGS_maxtsz);
  sampleCount_ = speechSamplesMetaInfo.size();
  sampleSizeOrder_ = sortSamples(
      speechSamplesMetaInfo,
      FLAGS_dataorder,
      FLAGS_inputbinsize,
      FLAGS_outputbinsize);

  shuffle(-1);
  LOG(INFO) << "Total batches (i.e. iters): " << sampleBatches_.size();
}

NoiseW2lListFilesDataset::~NoiseW2lListFilesDataset() {
  threadpool_ = nullptr; // join all threads
}

std::vector<af::array> NoiseW2lListFilesDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  W2lFeatureData feat;
  if (FLAGS_nthread > 0) {
    feat = getFeatureDataAndPrefetch(idx);
  } else {
    feat = getFeatureData(idx);
  }
  std::vector<af::array> result(kNumDataIdx);
  result[kInputIdx] = feat.input.empty()
      ? af::array(feat.inputDims)
      : af::array(feat.inputDims, feat.input.data());
  for (const auto& target : feat.targets) {
    auto targetType = target.first;
    auto targetData = target.second;
    auto targetDims = feat.targetDims[targetType];
    result[targetType] = targetData.empty()
        ? af::array(targetDims)
        : af::array(targetDims, targetData.data());
  }
  result[kSampleIdx] = feat.sampleIds.empty()
      ? af::array(feat.sampleIdsDims)
      : af::array(feat.sampleIdsDims, feat.sampleIds.data());
  return result;
}

W2lFeatureData NoiseW2lListFilesDataset::getFeatureData(
    const int64_t idx) const {
  auto ldData = getLoaderData(idx);
  return featurize2(ldData, dicts_);
}

W2lFeatureData NoiseW2lListFilesDataset::getFeatureDataAndPrefetch(
    const int64_t idx) const {
  W2lFeatureData feat;
  // check cache
  auto cachedata = prefetchCache_.find(idx);
  if (cachedata != prefetchCache_.end()) {
    feat = cachedata->second.get();
    prefetchCache_.erase(idx);
  } else {
    feat = getFeatureData(idx);
  }

  int64_t prefetchSize = FLAGS_nthread;
  // remove from cache (if necessary)
  for (auto it = prefetchCache_.begin(); it != prefetchCache_.end();) {
    if (it->first < idx || it->first > idx + prefetchSize) {
      it = prefetchCache_.erase(it);
      continue;
    } else {
      ++it;
    }
  }

  // add to cache
  for (int64_t i = idx + 1; i < std::min(idx + 1 + prefetchSize, size()); ++i) {
    if (prefetchCache_.find(i) == prefetchCache_.end()) {
      prefetchCache_.emplace(
          i,
          threadpool_->enqueue(
              [this](int64_t j) { return this->getFeatureData(j); }, i));
    }
  }
  return feat;
}

std::vector<W2lLoaderData> NoiseW2lListFilesDataset::getLoaderData(
    const int64_t idx) const {
  std::vector<W2lLoaderData> data(sampleBatches_[idx].size(), W2lLoaderData());
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "W2lListFilesDataset::getLoaderData idx out of range");
    }

    data[id].sampleId = data_[i].getSampleId();
    data[id].input = loadSound(data_[i].getAudioFile());
    data[id].targets[kTargetIdx] = wrd2Target(
        data_[i].getTranscript(),
        lexicon_,
        dicts_.at(kTargetIdx),
        fallback2Ltr_,
        skipUnk_);
    data[id].targets[kNoiseKeyIdx] = wrd2Target(
        data_[i].getTranscript(),
        lexicon_,
        dicts_.at(kNoiseKeyIdx),
        fallback2Ltr_,
        skipUnk_);

    // TODO use a flag to trigger that if needed.
    if (FLAGS_storeForNoiselm) {
      data[id].targets[kCleanNoiselmKeyIdx] = wrd2Target(
          data_[i].getGroundTruthTranscript(),
          lexicon_,
          dicts_.at(kCleanNoiselmKeyIdx),
          fallback2Ltr_,
          skipUnk_);
      data[id].targets[kNoisyNoiselmKeyIdx] = wrd2Target(
          data_[i].getTranscript(),
          lexicon_,
          dicts_.at(kNoisyNoiselmKeyIdx),
          fallback2Ltr_,
          skipUnk_);
    }

    if (includeWrd_) {
      data[id].targets[kWordIdx] = data_[i].getTranscript();
    }

    if (FLAGS_storeGroundTruth) {
      data[id].targets[kCleanKeyIdx] = wrd2Target(
          data_[i].getGroundTruthTranscript(),
          lexicon_,
          dicts_.at(kTargetIdx),
          fallback2Ltr_,
          skipUnk_);
    }
  }
  return data;
}

void NoiseW2lListFilesDataset::eraseTargets(const int64_t idx) {
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "W2lListFilesDataset::eraseTargets idx out of range");
    }
    std::vector<std::string> emptyTranscript = {};
    data_[i].setTranscript(emptyTranscript);
  }
}

void NoiseW2lListFilesDataset::updateTargets(
    const int64_t idx,
    const std::vector<std::vector<std::string>>& newTranscriptions,
    bool toGroundTruth) {
  if (sampleBatches_[idx].size() != newTranscriptions.size()) {
    throw std::out_of_range(
        "W2lListFilesDataset::updateTargets batch size is not the same for the sample and the provided new transcriptions");
  }

  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "W2lListFilesDataset::updateTargets idx out of range");
    }
    if (toGroundTruth) {
      data_[i].setGroundTruthTranscript(newTranscriptions[id]);
    } else {
      data_[i].setTranscript(newTranscriptions[id]);
    }
  }
}

void NoiseW2lListFilesDataset::copyToGroundTruthTranscript(const int64_t idx) {
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "W2lListFilesDataset::updateTargets idx out of range");
    }
    data_[i].setGroundTruthTranscript(data_[i].getTranscript());
  }
}

std::vector<float> NoiseW2lListFilesDataset::loadSound(
    const std::string& audioHandle) const {
  return w2l::loadSound<float>(audioHandle);
}

std::vector<SpeechSampleMetaInfo> NoiseW2lListFilesDataset::loadListFile(
    const std::string& filename) {
  std::ifstream infile(filename);

  LOG_IF(FATAL, !infile) << "Could not read file '" << filename << "'";

  // The format of the list: columns should be space-separated
  // [utterance id] [audio file (full path)] [audio length] [word transcripts]
  std::string line;
  std::vector<SpeechSampleMetaInfo> samplesMetaInfo;
  auto curDataSize = data_.size();
  int64_t idx = curDataSize;
  while (std::getline(infile, line)) {
    // tab separated mode. May contain 0 or 1 target or 1 target + 1 ground
    // truth
    if (line.find("\t") != std::string::npos) {
      double audioLength;
      auto tokens_tab = splitOnAnyOf(
          "\t", line, false); // we check if the ground truth is given
      audioLength = std::stod(tokens_tab[2]);

      if (tokens_tab.size() == 5) {
        // LOG_IF(FATAL, tokens_tab.size() < 5) << "Cannot parse, ground truth
        // not given " << line; LOG(INFO) << "Noisy transcriptions and ground
        // truth provided for " << filename;
        auto train_trans = splitOnWhitespace(tokens_tab[3], true);
        auto train_truth = splitOnWhitespace(tokens_tab[4], true);

        data_.emplace_back(SpeechSample2(
            tokens_tab[0],
            tokens_tab[1],
            std::vector<std::string>(train_trans.begin(), train_trans.end()),
            std::vector<std::string>(train_truth.begin(), train_truth.end())));

      } else if (tokens_tab.size() == 4) {
        // LOG(INFO) << "1 transcription provided for " << filename;
        auto train_trans = splitOnWhitespace(tokens_tab[3], true);

        // LOG_IF(FATAL, tokens.size() < 4) << "Cannot parse " << line;

        data_.emplace_back(SpeechSample2(
            tokens_tab[0],
            tokens_tab[1],
            std::vector<std::string>(train_trans.begin(), train_trans.end())));

      } else if (tokens_tab.size() == 3) { // unsupervised data

        data_.emplace_back(SpeechSample2(tokens_tab[0], tokens_tab[1], {}));

      } else {
        LOG_IF(FATAL, true) << "Cannot parse " << filename;
      }

      auto targets = wrd2Target(
          data_.back().getTranscript(),
          lexicon_,
          dicts_.at(kTargetIdx),
          fallback2Ltr_,
          skipUnk_);

      samplesMetaInfo.emplace_back(
          SpeechSampleMetaInfo(audioLength, targets.size(), idx));

      ++idx;
    } else {
      // Normal mode. 1 target transcription
      auto tokens = splitOnWhitespace(line, true);

      LOG_IF(FATAL, tokens.size() < 3) << "Cannot parse " << line;

      data_.emplace_back(SpeechSample2(
          tokens[0],
          tokens[1],
          std::vector<std::string>(tokens.begin() + 3, tokens.end())));

      auto audioLength = std::stod(tokens[2]);
      auto targets = wrd2Target(
          data_.back().getTranscript(),
          lexicon_,
          dicts_.at(kTargetIdx),
          fallback2Ltr_,
          skipUnk_);

      samplesMetaInfo.emplace_back(
          SpeechSampleMetaInfo(audioLength, targets.size(), idx));

      ++idx;
    }
  }

  LOG(INFO) << samplesMetaInfo.size() << " files found. ";

  return samplesMetaInfo;
}

} // namespace w2l
