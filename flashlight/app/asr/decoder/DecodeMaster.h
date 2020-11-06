/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/dataset/datasets.h>
#include "flashlight/fl/nn/nn.h"
// #include "flashlight/lib/common/WordUtils.h"??
#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace fl {
namespace app {
namespace asr {

struct DecodeMasterLexiconFreeOptions {
  int beamSize;
  int beamSizeToken;
  float beamThreshold;
  float lmWeight;
  float silScore;
  bool logAdd;
  std::string silToken;
  std::string blankToken;
};

struct DecodeMasterLexiconOptions {
  int beamSize;
  int beamSizeToken;
  float beamThreshold;
  float lmWeight;
  float silScore;
  float wordScore;
  float unkScore;
  bool logAdd;
  std::string silToken;
  std::string blankToken;
  std::string unkToken;
  w2l::SmearingMode smearMode;
};

struct TrainOptions {
  int replabel, bool useWordPiece, std::string surround,
};

class DecodeMaster {
 public:
  explicit DecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      bool isTokenLM,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict,
      const TrainOptions trainOpt);

  // compute emissions
  virtual std::shared_ptr<fl::Dataset> forward(
      const std::shared_ptr<fl::Dataset>& ds);

  // decode emissions with an existing decoder
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      fl::lib::text::Decoder& decoder);

  std::tuple<std::vector<double>, std::vector<double>> computeLERWER(
      const std::shared_ptr<fl::Dataset>& pds);

  // post process predictions
  virtual std::vector<int> postProcessPreds(
      const std::vector<int>& tokenIdxSeq) = 0;

  // post process target prediction
  virtual std::vector<int> postProcessTarget(
      const std::vector<int>& tokenIdxSeq) = 0;

 protected:
  std::shared_ptr<fl::lib::text::Trie> buildTrie(
      const fl::lib::text::LexiconMap& lexicon,
      std::string silToken,
      fl::lib::text::SmearingMode smearMode,
      int replabel) const;

  std::shared_ptr<fl::Module> net_;
  std::shared_ptr<fl::lib::text::LM> lm_;
  bool isTokenLM_;
  fl::lib::text::Dictionary tokenDict_;
  fl::lib::text::Dictionary wordDict_;
  trainOptions trainOpt_;
};

class TokenDecodeMaster : public DecodeMaster {
 public:
  explicit TokenDecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict);

  // compute predictions from emissions
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      DecodeMasterLexiconFreeOptions opt);

  // compute predictions from emissions
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      const fl::lib::text::LexiconMap& lexicon,
      DecodeMasterLexiconOptions opt);

  // post process predictions
  std::vector<int> postProcessPreds(const std::vector<int>& tokenIdxSeq);

  // post process target prediction
  std::vector<int> postProcessTarget(const std::vector<int>& tokenIdxSeq);
};

class WordDecodeMaster : public DecodeMaster {
 public:
  explicit WordDecodeMaster(
      const std::shared_ptr<fl::Module> net,
      const std::shared_ptr<fl::lib::text::LM> lm,
      const fl::lib::text::Dictionary& tokenDict,
      const fl::lib::text::Dictionary& wordDict);

  // compute predictions from emissions
  std::shared_ptr<fl::Dataset> decode(
      const std::shared_ptr<fl::Dataset>& eds,
      const fl::lib::text::LexiconMap& lexicon,
      DecodeMasterLexiconOptions opt);

  // post process predictions
  std::vector<int> postProcessPreds(const std::vector<int>& tokenIdxSeq);

  // post process target prediction
  std::vector<int> postProcessTarget(const std::vector<int>& tokenIdxSeq);
};
} // namespace asr
} // namespace app
} // namespace fl
