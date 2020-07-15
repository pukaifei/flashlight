/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <queue>

//#include "experimental/lead2Gold/src/criterion/l2g/EncDecCriterion.cpp"
#include "experimental/lead2Gold/src/criterion/l2g/EncDecCriterionMulti.h"

using namespace fl;

namespace w2l {

namespace detailMulti {

EDState concatState(std::vector<EDState>& stateVec) {
  if (stateVec.size() < 1) {
    throw std::runtime_error("Empty stateVec");
  }

  int nbLayer = stateVec[0].hidden.size();
  EDState newState(nbLayer);
  newState.step = stateVec[0].step;
  newState.isValid = stateVec[0].isValid;

  std::vector<std::vector<Variable>> hiddenVec(nbLayer);
  for (auto& state : stateVec) {
    if (state.step != newState.step) {
      throw std::runtime_error("step unmatched");
    } else if (state.isValid != newState.isValid) {
      throw std::runtime_error("isValid unmatched");
    }
    for (int i = 0; i < nbLayer; i++) {
      hiddenVec[i].push_back(state.hidden[i]);
    }
  }
  for (int i = 0; i < nbLayer; i++) {
    newState.hidden[i] = concatenate(hiddenVec[i], 2); ///// <--- I changed that
  }
  return newState;
}

EDState selectState(EDState& state, int batchIdx) {
  int nbLayer = state.hidden.size();
  EDState newState(nbLayer);
  newState.step = state.step;
  newState.isValid = state.isValid;
  for (int i = 0; i < nbLayer; i++) {
    newState.hidden[i] =
        state.hidden[i](af::span, af::span, batchIdx); ///// <--- I changed that
  }
  return newState;
}
} // namespace detailMulti

EncDecCriterionMulti::EncDecCriterionMulti(
    int nClass,
    int modelDimInX,
    int modelDimOut,
    int eos,
    int maxDecoderOutputLen,
    int nLayerEncT,
    int nLayerDec,
    double labelSmooth,
    double pctTeacherForcing,
    double p_dropout,
    double p_layerdrop,
    bool useSinPosEmb,
    bool posEmbEveryLayer)
    : nClass_(nClass),
      eos_(eos),
      maxDecoderOutputLen_(maxDecoderOutputLen),
      nLayerEncT_(nLayerEncT),
      nLayerDec_(nLayerDec),
      labelSmooth_(labelSmooth),
      pctTeacherForcing_(pctTeacherForcing),
      useSinPosEmb_(useSinPosEmb),
      posEmbEveryLayer_(posEmbEveryLayer) {
  add(std::make_shared<fl::Embedding>(modelDimOut, nClass));
  for (size_t i = 0; i < nLayerEncT_; i++) {
    bool useTrPos =
        useSinPosEmb ? false : (posEmbEveryLayer_ || i == 0 ? true : false);

    add(std::make_shared<TransformerBlockSimple>(
        modelDimOut,
        modelDimOut / 4,
        modelDimOut * 4,
        4,
        maxDecoderOutputLen,
        p_dropout,
        p_layerdrop,
        useTrPos));
  }
  for (size_t i = 0; i < nLayerDec_; i++) {
    bool useTrPos =
        useSinPosEmb ? false : (posEmbEveryLayer_ || i == 0 ? true : false);
    // DIM IN 1 = Transcription attention
    // DIM IN 2 = X attention
    add(std::make_shared<w2l::TransformerBlockMultiAttend>(
        modelDimOut,
        modelDimInX,
        modelDimOut,
        modelDimOut * 4,
        4,
        maxDecoderOutputLen,
        p_dropout,
        p_layerdrop,
        useTrPos));
  }

  add(std::make_shared<fl::Linear>(modelDimOut, nClass));
  params_.push_back(
      fl::uniform(af::dim4{modelDimOut}, -1e-1, 1e-1)); // for initial embedding

  if (useSinPosEmb_) {
    int maxLen = maxDecoderOutputLen + 200; // take some marge
    std::vector<float> embFlat(maxLen * modelDimOut);
    for (int pos = 0; pos < maxLen; pos++) {
      for (int i = 0; i < modelDimOut; i++) {
        if (i % 2 == 0) {
          embFlat[i + modelDimOut * pos] = std::sin(
              (float)pos /
              (std::pow(10000.0, ((float)i / (float)modelDimOut))));
        } else {
          embFlat[i + modelDimOut * pos] = std::cos(
              (float)pos /
              (std::pow(10000.0, ((float)(i - 1) / (float)modelDimOut))));
        }
      }
    }
    sinPosEmb =
        fl::Variable(af::array(modelDimOut, maxLen, embFlat.data()), false);
  } else {
    sinPosEmb = fl::Variable();
  }
}

// inputs has to contain input and target sentences.
std::vector<Variable> EncDecCriterionMulti::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 3) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& inputT = inputs[0];
  const auto& inputX = inputs[1];
  const auto& target = inputs.back();

  auto encodedTranscript = encodeT(inputT);

  auto out = vectorizedDecoder({encodedTranscript, inputX}, target).front();

  out = logSoftmax(out, 0);

  auto losses = moddims(
      sum(categoricalCrossEntropy(out, target, ReduceMode::NONE), {0}), -1);
  if (train_ && labelSmooth_ > 0) {
    size_t nClass = out.dims(0);
    auto smoothLoss = moddims(sum(out, {0, 1}), -1);
    losses = (1 - labelSmooth_) * losses - (labelSmooth_ / nClass) * smoothLoss;
  }

  return {losses, out};
}

// input of size  D x T x B
Variable EncDecCriterionMulti::applyPosEmb(
    const Variable& input,
    const int offset = 0) const {
  if (useSinPosEmb_) {
    fl::Variable posEmb = tile(
        sinPosEmb.cols(offset, offset + (int)input.dims(1) - 1),
        af::dim4(1, 1, (int)input.dims(2)));
    return input + posEmb;
  } else {
    return input;
  }
}

// inputT should be of size U * B
Variable EncDecCriterionMulti::encodeT(const Variable& inputT) {
  auto encoded = embedding()->forward(inputT);
  encoded = applyPosEmb(encoded);
  for (size_t i = 0; i < nLayerEncT_; i++) {
    encoded = layerEncT(i)->forward({encoded}).front();
  }
  return encoded;
}

// encoded[0] : D1 x T1 x B, encoded[1] : D2 x T2 x B
// target: U x B

std::vector<Variable> EncDecCriterionMulti::vectorizedDecoder(
    const std::vector<Variable>& encoded,
    const Variable& target) {
  int U = target.dims(0);
  int B = target.dims(1);
  // int T = encoded.isempty() ? 0 : encoded.dims(1);

  auto hy = tile(startEmbedding(), {1, 1, B});

  if (U > 1) {
    auto y = target(af::seq(0, U - 2), af::span); // remove last token

    if (train_) {
      // TODO: other sampling strategies
      auto mask =
          Variable(af::randu(y.dims()) * 100 <= pctTeacherForcing_, false);
      auto samples =
          Variable((af::randu(y.dims()) * (nClass_ - 1)).as(s32), false);
      y = mask * y + (1 - mask) * samples;
    }

    auto yEmbed = embedding()->forward(y);
    hy = concatenate({hy, yEmbed}, 1); // Add one initial embedding
  }

  if (!posEmbEveryLayer_) {
    hy = applyPosEmb(hy);
  }
  // Variable alpha, summaries;
  for (int i = 0; i < nLayerDec_; i++) {
    if (posEmbEveryLayer_) {
      hy = applyPosEmb(hy);
    }
    hy = layerDec(i)->forward(std::vector<Variable>({hy}), encoded).front();
  }

  auto out = linearOut()->forward(hy);

  return {out};
}

af::array EncDecCriterionMulti::viterbiPath(
    const std::vector<Variable>& encoded) {
  return viterbiPathBase(encoded);
}

af::array EncDecCriterionMulti::viterbiPathBase(
    const std::vector<Variable>& encoded,
    bool inc_eos) {
  bool wasTrain = train_;
  eval();
  std::vector<int> path;
  EDState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;

  for (int u = 0; u < maxDecoderOutputLen_; u++) {
    // ox = decodeStep(Variable(encoded, false), y);
    std::tie(ox, state) = decodeStep(encoded, y, state);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);

    if (pred == eos_) {
      if (inc_eos) {
        path.push_back(pred);
      }
      break;
    }
    y = constant(pred, 1, s32, false);
    path.push_back(pred);
  }

  if (wasTrain) {
    train();
  }

  auto vPath = path.empty() ? af::array() : af::array(path.size(), path.data());
  return vPath;
}

af::array EncDecCriterionMulti::viterbiCheat(
    const std::vector<Variable>& encoded,
    const af::array& cleanTarget,
    bool inc_eos) {
  bool wasTrain = train_;
  eval();
  std::vector<int> path;
  EDState state;
  Variable y, ox;
  af::array maxIdx, maxValues;
  int pred;

  int maxDecode = std::min(maxDecoderOutputLen_, (int)cleanTarget.dims(0));
  for (int u = 0; u < maxDecode; u++) {
    // ox = decodeStep(Variable(encoded, false), y);
    std::tie(ox, state) = decodeStep(encoded, y, state);
    max(maxValues, maxIdx, ox.array());
    maxIdx.host(&pred);

    if (pred == eos_) {
      if (inc_eos) {
        path.push_back(pred);
      }
      break;
    }
    y = fl::Variable(cleanTarget(u), false);
    path.push_back(pred);
  }

  if (wasTrain) {
    train();
  }

  auto vPath = path.empty() ? af::array() : af::array(path.size(), path.data());
  return vPath;
}

std::pair<Variable, EDState> EncDecCriterionMulti::decodeStep(
    const std::vector<Variable>& encoded,
    const Variable& y,
    const EDState& inState) const {
  Variable hy;
  if (y.isempty()) {
    hy = tile(startEmbedding(), {1, 1, encoded[0].dims(2)});
  } else {
    hy = embedding()->forward(y);
  }

  if (!posEmbEveryLayer_) {
    hy = applyPosEmb(hy, inState.step);
  }

  EDState outState;
  outState.step = inState.step + 1;
  for (int i = 0; i < nLayerDec_; i++) {
    if (posEmbEveryLayer_) {
      hy = applyPosEmb(hy, inState.step);
    }
    if (inState.step == 0) {
      outState.hidden.push_back(hy);
      hy = layerDec(i)->forward(std::vector<Variable>({hy}), encoded).front();
    } else {
      auto tmp = std::vector<Variable>({inState.hidden[i], hy});
      outState.hidden.push_back(concatenate(tmp, 1));
      hy = layerDec(i)->forward(tmp, encoded).front();
    }
  }

  auto out = linearOut()->forward(hy);
  return std::make_pair(out, outState);
}

std::vector<int> EncDecCriterionMulti::beamPath(
    const std::vector<Variable>& encoded,
    int beamSize /* = 10 */,
    float eos_score) {
  std::vector<EncDecCriterionMulti::CandidateHypo> beam;
  auto ini_state = EDState(nLayerDec_);
  auto ini_candidate = CandidateHypo(0, {}, ini_state);
  // beam.emplace_back(CandidateHypo{});
  beam.emplace_back(ini_candidate);
  auto beamPaths =
      beamSearch(encoded, beam, beamSize, maxDecoderOutputLen_, eos_score);
  return beamPaths[0].path;
}

std::vector<EncDecCriterionMulti::CandidateHypo>
EncDecCriterionMulti::beamSearchRes(
    const std::vector<Variable>& encoded,
    int beamSize /* = 10 */,
    float eos_score) {
  std::vector<EncDecCriterionMulti::CandidateHypo> beam;
  auto ini_state = EDState(nLayerDec_);
  auto ini_candidate = CandidateHypo(0, {}, ini_state);
  // beam.emplace_back(CandidateHypo{});
  beam.emplace_back(ini_candidate);
  return beamSearch(encoded, beam, beamSize, maxDecoderOutputLen_, eos_score);
}

// beam are candidates that need to be extended
std::vector<EncDecCriterionMulti::CandidateHypo>
EncDecCriterionMulti::beamSearch(
    const std::vector<Variable>& encoded, // H x T x 1
    std::vector<EncDecCriterionMulti::CandidateHypo> beam,
    int beamSize = 10,
    int maxLen = 200,
    float eos_score = 0) {
  bool wasTrain = train_;
  eval();

  std::vector<EncDecCriterionMulti::CandidateHypo> complete;
  std::vector<EncDecCriterionMulti::CandidateHypo> newBeam;
  auto cmpfn = [](EncDecCriterionMulti::CandidateHypo& lhs,
                  EncDecCriterionMulti::CandidateHypo& rhs) {
    return lhs.score > rhs.score;
  };

  for (int l = 0; l < maxLen; l++) {
    // std::cout << "l= " << l << std::endl;
    newBeam.resize(0);

    std::vector<Variable> prevYVec;
    std::vector<EDState> prevStateVec;
    std::vector<float> prevScoreVec;
    // int j=0;
    for (auto& hypo : beam) {
      //++j;
      // std::cout << "state num: " << j << std::endl;
      Variable y;
      if (!hypo.path.empty()) {
        y = constant(hypo.path.back(), 1, s32, false);
      }
      prevYVec.push_back(y);
      // af::print("push y: ", y.array());
      prevStateVec.push_back(hypo.state);
      // af::print("push state: ", hypo.state.array());
      prevScoreVec.push_back(hypo.score);
    }
    auto prevY = concatenate(prevYVec, 1); // 1 x B
    // af::print("prevY: ", prevY.array());
    // std::cout << "call concatState " << std::endl;
    auto prevState = detailMulti::concatState(prevStateVec);

    Variable ox;
    EDState state;
    // std::tie(ox, state) = decodeStep(Variable(input, false), prevY,
    // prevState);
    std::tie(ox, state) = decodeStep(
        {tile(encoded[0], {1, 1, static_cast<int>(beam.size())}),
         tile(encoded[1], {1, 1, static_cast<int>(beam.size())})},
        prevY,
        prevState);
    ox = logSoftmax(ox, 0); // C x 1 x B
    ox = fl::reorder(ox, 0, 2, 1);

    auto scoreArr =
        af::array(1, static_cast<int>(beam.size()), prevScoreVec.data());
    scoreArr = af::tile(scoreArr, ox.dims()[0]);

    scoreArr = scoreArr + ox.array(); // C x B
    scoreArr = af::flat(scoreArr); // column-first
    auto scoreVec = w2l::afToVector<float>(scoreArr);

    std::vector<size_t> indices(scoreVec.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(
        indices.begin(),
        indices.begin() +
            std::min(2 * beamSize, static_cast<int>(scoreVec.size())),
        indices.end(),
        [&scoreVec](size_t i1, size_t i2) {
          return scoreVec[i1] > scoreVec[i2];
        });

    int nClass = ox.dims()[0];
    for (int j = 0; j < indices.size(); j++) {
      int hypIdx = indices[j] / nClass;
      int clsIdx = indices[j] % nClass;
      std::vector<int> path_(beam[hypIdx].path);
      path_.push_back(clsIdx);
      if (j < beamSize && clsIdx == eos_) {
        // path_.pop_back();
        complete.emplace_back(
            scoreVec[indices[j]] + eos_score,
            path_,
            detailMulti::selectState(state, hypIdx));
      } else if (clsIdx != eos_) {
        newBeam.emplace_back(
            scoreVec[indices[j]],
            path_,
            detailMulti::selectState(state, hypIdx));
      }
      if (newBeam.size() >= beamSize) {
        break;
      }
    }
    beam.resize(newBeam.size());
    beam = std::move(newBeam);

    if (complete.size() >= beamSize) {
      std::partial_sort(
          complete.begin(), complete.begin() + beamSize, complete.end(), cmpfn);
      complete.resize(beamSize);

      // if lowest score in complete is better than best future hypo
      // then its not possible for any future hypothesis to replace existing
      // hypothesises in complete.
      if (complete.back().score > beam[0].score) {
        break;
      }
    }
  }

  if (wasTrain) {
    train();
  }

  return complete.empty() ? beam : complete;
}

/*
std::pair<std::vector<std::vector<float>>, std::vector<TS2SStatePtr>>
TransformerCriterion::decodeBatchStep(
    const fl::Variable& xEncoded,
    std::vector<fl::Variable>& ys,
    const std::vector<TS2SState*>& inStates,
    //const int ,
    const float smoothingTemperature) const {
  size_t stepSize = fl::afGetMemStepSize();
  fl::afSetMemStepSize(10 * (1 << 10));
  int B = ys.size();

  for (int i = 0; i < B; i++) {
    if (ys[i].isempty()) {
      ys[i] = startEmbedding();
    } else {
      ys[i] = embedding()->forward(ys[i]);
    } // TODO: input feeding
    ys[i] = moddims(ys[i], {ys[i].dims(0), 1, -1});
  }
  Variable yBatched = concatenate(ys, 2); // D x 1 x B

  std::vector<TS2SStatePtr> outstates(B);
  for (int i = 0; i < B; i++) {
    outstates[i] = std::make_shared<TS2SState>();
    outstates[i]->step = inStates[i]->step + 1;
  }

  Variable outStateBatched;
  for (int i = 0; i < nLayer_; i++) {
    if (inStates[0]->step == 0) {
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(yBatched.slice(j));
      }
      yBatched = layer(i)->forward(std::vector<Variable>({yBatched})).front();
    } else {
      std::vector<Variable> statesVector(B);
      for (int j = 0; j < B; j++) {
        statesVector[j] = inStates[j]->hidden[i];
      }
      Variable inStateHiddenBatched = concatenate(statesVector, 2);
      auto tmp = std::vector<Variable>({inStateHiddenBatched, yBatched});
      auto tmp2 = concatenate(tmp, 1);
      for (int j = 0; j < B; j++) {
        outstates[j]->hidden.push_back(tmp2.slice(j));
      }
      yBatched = layer(i)->forward(tmp).front();
    }
  }

  Variable alpha, summary;
  yBatched = moddims(yBatched, {yBatched.dims(0), -1});
  std::tie(alpha, summary) =
      attention()->forward(yBatched, xEncoded, Variable(), Variable());
  alpha = reorder(alpha, 1, 0);
  yBatched = yBatched + summary;

  auto outBatched = linearOut()->forward(yBatched);
  outBatched = logSoftmax(outBatched / smoothingTemperature, 0);
  std::vector<std::vector<float>> out(B);
  for (int i = 0; i < B; i++) {
    out[i] = w2l::afToVector<float>(outBatched.col(i));
  }

  fl::afSetMemStepSize(stepSize);
  return std::make_pair(out, outstates);
}
*/

/*
AMUpdateFunc buildTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& c) {
  auto buf = std::make_shared<TS2SDecoderBuffer>(
      FLAGS_beamsize, FLAGS_attentionthreshold, FLAGS_smoothingtemperature);

  const TransformerCriterion* criterion =
      static_cast<TransformerCriterion*>(c.get());

  auto amUpdateFunc = [buf, criterion](
                          const float* emissions,
                          const int N,
                          const int T,
                          const std::vector<int>& rawY,
                          const std::vector<AMStatePtr>& rawPrevStates,
                          int& t) {
    if (t == 0) {
      buf->input = fl::Variable(af::array(N, T, emissions), false);
    }
    int B = rawY.size();
    buf->prevStates.resize(0);
    buf->ys.resize(0);

    for (int i = 0; i < B; i++) {
      TS2SState* prevState = static_cast<TS2SState*>(rawPrevStates[i].get());
      fl::Variable y;
      if (t > 0) {
        y = fl::constant(rawY[i], 1, s32, false);
      } else {
        prevState = &buf->dummyState;
      }
      buf->ys.push_back(y);
      buf->prevStates.push_back(prevState);
    }

    std::vector<std::vector<float>> amScores;
    std::vector<TS2SStatePtr> outStates;

    std::tie(amScores, outStates) = criterion->decodeBatchStep(
        buf->input,
        buf->ys,
        buf->prevStates,
        buf->attentionThreshold,
        buf->smoothingTemperature);

    std::vector<AMStatePtr> out;
    for (auto& os : outStates) {
      out.push_back(os);
    }

    return std::make_pair(amScores, out);
  };

  return amUpdateFunc;
}
*/

std::string EncDecCriterionMulti::prettyString() const {
  return "EncDecCriterionMulti";
}

} // namespace w2l
