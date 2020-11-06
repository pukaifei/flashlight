/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/app/asr/decoder/DecodeMaster.h"

#include "flashlight/ext/common/Utils-int.h"
#include "flashlight/fl/app/asr/common/Defines.h"
#include "flashlight/fl/dataset/MemoryBlobDataset.h"
#include "flashlight/fl/lib/text/decoder/Decoder.h"
#include "flashlight/fl/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/fl/lib/text/decoder/LexiconFreeDecoder.h"

namespace {
af::array removeNegative(const af::array& arr) {
  return arr(arr >= 0);
}
} // namespace
namespace fl {
namespace app {
namespace asr {

DecodeMaster::DecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    bool isTokenLM,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict,
    const trainOptions trainOpt)
    : net_(net),
      lm_(lm),
      isTokenLM_(isTokenLM),
      tokenDict_(tokenDict),
      wordDict_(wordDict),
      trainOpt_(trainOpt) {}

std::tuple<std::vector<double>, std::vector<double>>
DecodeMaster::computeLERWER(const std::shared_ptr<fl::Dataset>& pds) {
  fl::EditDistanceMeter wer, ler;
  for (auto& sample : *pds) {
    if (sample.size() <= kTokenIdx) {
      throw std::runtime_error(
          "computeLERWER: need word target to compute WER");
    }
    auto prediction = sample[kInputIdx];
    auto target = sample[kTokenIdx];
    if (prediction.numdims() > 2 || target.numdims() > 2) {
      throw std::runtime_error(
          "computeLERWER: expecting TxB for prediction and target");
    }
    if (!prediction.isempty() && !target.isempty() &&
        (prediction.dims(1) != target.dims(1))) {
      throw std::runtime_error(
          "computeLERWER: prediction and target do not match");
    }
    // token predictions and target
    std::vector<int> predictionV = afToVector<int>(prediction);
    std::vector<int> targetV = afToVector<int>(target);
    predictionV = postProcessPred(predictionV);
    targetV = postProcessTarget(targetV);

    std::vector<std::string> predictionS =
        tknIdx2Ltr(predictionV, tokenDict_, silToken);
    std::vector<std::string> targetS =
        tknIdx2Ltr(targetV, tokenDict_, silToken);
    ler.add(predictionS, targetS);
    wer.add(tkn2Wrd(predictionS), tkn2Wrd(targetS));
  }
  return {ler.value(), wer.value()};
}

std::vector<std::string> postProcessTarget(const std::vector<int>& tknIdxSeq) {}

std::shared_ptr<Trie> DecodeMaster::buildTrie(
    const fl::lib::text:: ::LexiconMap& lexicon,
    std::string silToken,
    fl::lib::text::SmearingMode smearMode,
    int replabel) const {
  auto trie = std::make_shared<Trie>(
      tokenDict_.indexSize(), tokenDict_.getIndex(silToken));
  auto startState = lm_->start(false);
  for (auto& it : lexicon) {
    const std::string& word = it.first;
    int usrIdx = wordDict_.getIndex(word);
    float score = 0;
    if (!isTokenLM_) {
      LMStatePtr dummyState;
      std::tie(dummyState, score) = lm_->score(startState, usrIdx);
    }
    for (auto& tokens : it.second) {
      auto tokensTensor = tkn2Idx(tokens, tokenDict_, replabel);
      trie->insert(tokensTensor, usrIdx, score);
    }
  }
  // Smearing
  trie->smear(smearMode);
  return trie;
}

std::shared_ptr<fl::Dataset> DecodeMaster::forward(
    const std::shared_ptr<fl::Dataset>& ds) {
  auto eds = std::make_shared<fl::MemoryBlobDataset>();
  for (auto& batch : *ds) {
    auto output = net_->forward({fl::input(batch[kInputIdx])}).front().array();
    if (output.numdims() > 3) {
      throw std::runtime_error("output should be NxTxB");
    }
    af::array tokenTarget =
        (batch.size() > kTargetIdx ? batch[kTargetIdx] : af::array());
    af::array wordTarget =
        (batch.size() > kWordIdx ? batch[kWordIdx] : af::array());

    int B = output.dims(2);
    if (!tokenTarget.isempty() &&
        (tokenTarget.numdims() > 2 || tokenTarget.dims(1) != B)) {
      throw std::runtime_error("token target should be LxB");
    }
    if (!wordTarget.isempty() &&
        (wordTarget.numdims() > 2 || wordTarget.dims(1) != B)) {
      throw std::runtime_error("word target should be LxB");
    }
    // todo s2s
    for (int b = 0; b < B; b++) {
      std::vector<af::array> res(3);
      res[kInputIdx] = output(af::span, af::span, b);
      res[kTargetIdx] = removeNegative(tokenTarget(af::span, b));
      res[kWordIdx] = removeNegative(wordTarget(af::span, b));
      eds->add(res);
    }
  }
  eds->writeIndex();
  return eds;
}

// threading?
// wer on tokens, what to do in case of lexfree
// cleaning predicitons?
// replabel -> <1>
// fix wer from tokes, squeeze in each decoder

std::shared_ptr<fl::Dataset> DecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& eds,
    fl::lib::text::Decoder& decoder) {
  auto pds = std::make_shared<fl::MemoryBlobDataset>();
  for (auto& sample : *eds) {
    auto emission = sample[kInputIdx];
    if (emission.numdims() > 2) {
      throw std::runtime_error("emission should be NxT");
    }
    std::vector<float> emissionV(emission.elements());
    emission.as(af::dtype::f32).host(emissionV.data());
    auto results =
        decoder.decode(emissionV.data(), emission.dims(1), emission.dims(0));
    std::vector<int> wordsV = results.at(0).words;
    std::vector<int> tokensV = results.at(0).tokens;
    wordsV.erase(std::remove(wordsV.begin(), wordsV.end(), -1), wordsV.end());
    sample[kInputIdx] =
        (wordsV.size() > 0 ? af::array(af::dim4(wordsV.size()), wordsV.data())
                           : af::array());
    pds->add(sample);
  }
  pds->writeIndex();
  return pds;
}

TokenDecodeMaster::TokenDecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict)
    : DecodeMaster(net, lm, true, tokenDict, wordDict) {}

std::shared_ptr<fl::Dataset> TokenDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& eds,
    DecodeMasterLexiconFreeOptions opt) {
  std::vector<float> transition;
  fl::lib::text::DecoderOptions decoderOpt(
      opt.beamSize,
      opt.beamSizeToken,
      opt.beamThreshold,
      opt.lmWeight,
      0,
      0,
      opt.silScore,
      0,
      opt.logAdd,
      CriterionType::CTC);
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  fl::lib::text::LexiconFreeDecoder decoder(
      decoderOpt, lm_, silIdx, blankIdx, transition);
  return DecodeMaster::decode(eds, decoder);
}

std::shared_ptr<fl::Dataset> TokenDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& eds,
    const fl::lib::text::LexiconMap& lexicon,
    fl::lib::text::DecodeMasterLexiconOptions opt) {
  std::vector<float> transition;
  auto trie = buildTrie(lexicon, opt.silToken, opt.smearMode, opt.repLabel);
  fl::lib::text::DecoderOptions decoderOpt(
      opt.beamSize,
      opt.beamSizeToken,
      opt.beamThreshold,
      opt.lmWeight,
      0,
      0,
      opt.silScore,
      0,
      opt.logAdd,
      CriterionType::CTC);
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  auto unkWordIdx = -1; // wordDict.getIndex(kUnkToken);
  fl::lib::text::LexiconDecoder decoder(
      decoderOpt, trie, lm_, silIdx, blankIdx, unkWordIdx, transition, true);
  return DecodeMaster::decode(eds, decoder);
}

WordDecodeMaster::WordDecodeMaster(
    const std::shared_ptr<fl::Module> net,
    const std::shared_ptr<fl::lib::text::LM> lm,
    const fl::lib::text::Dictionary& tokenDict,
    const fl::lib::text::Dictionary& wordDict)
    : DecodeMaster(net, lm, false, tokenDict, wordDict) {}

std::shared_ptr<fl::Dataset> WordDecodeMaster::decode(
    const std::shared_ptr<fl::Dataset>& eds,
    const fl::lib::text::LexiconMap& lexicon,
    fl::lib::text::DecodeMasterLexiconOptions opt) {
  std::vector<float> transition;
  auto trie = buildTrie(lexicon, opt.silToken, opt.smearMode, opt.repLabel);
  fl::lib::text::DecoderOptions decoderOpt(
      opt.beamSize,
      opt.beamSizeToken,
      opt.beamThreshold,
      opt.lmWeight,
      opt.wordScore,
      opt.unkScore,
      opt.silScore,
      0,
      opt.logAdd,
      CriterionType::CTC);
  auto silIdx = tokenDict_.getIndex(opt.silToken);
  auto blankIdx = tokenDict_.getIndex(opt.blankToken);
  auto unkWordIdx = wordDict_.getIndex(opt.unkToken);
  w2l::LexiconDecoder decoder(
      decoderOpt, trie, lm_, silIdx, blankIdx, unkWordIdx, transition, false);
  return DecodeMaster::decode(eds, decoder);
}

} // namespace asr
} // namespace app
} // namespace fl
