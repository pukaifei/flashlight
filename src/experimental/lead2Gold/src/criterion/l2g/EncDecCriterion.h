/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "common/FlashlightUtils.h"
#include "criterion/Defines.h"
#include "criterion/Seq2SeqCriterion.h"
#include "criterion/SequenceCriterion.h"
//#include "criterion/attention/attention.h"
//#include "criterion/attention/window.h"

//#include "flashlight/contrib/modules/Transformer.h"
#include "TransformerBlockSimple.h"
#include "TransformerBlockAttend.h"

namespace w2l {

struct EDState {
  //fl::Variable alpha;
  std::vector<fl::Variable> hidden;
  //fl::Variable summary;
  int step;
  //int peakAttnPos;
  bool isValid;

  EDState() : step(0), isValid(false) {}
  explicit EDState(int nbLayer)
      : hidden(nbLayer), step(0), isValid(false) {}
};

typedef std::shared_ptr<EDState> EDStatePtr;

class EncDecCriterion : public SequenceCriterion {
 public:
  struct CandidateHypo {
    float score;
    std::vector<int> path;
    EDState state;
    explicit CandidateHypo() : score(0.0) {
      path.resize(0);
    };
    CandidateHypo(float score_, std::vector<int> path_, EDState state_)
        : score(score_), path(path_), state(state_) {}
  };

  EncDecCriterion(
      int nClass,
      int hiddenDim,
      int eos,
      int maxDecoderOutputLen,
      int nLayerEnc,
      int nLayerDec,
      double labelSmooth,
      double pctTeacherForcing,
      double p_dropout,
      double p_layerdrop,
      bool useSinPosEmb,
      bool posEmbEveryLayer);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(const af::array& input) override;

  af::array viterbiPathBase(
      const af::array& encoded,
      bool inc_eos = false);

  af::array viterbiCheat(
    const af::array& encoded,
    const af::array& cleanTarget,
    bool inc_eos = false);

  std::vector<fl::Variable> encode(
      const std::vector<fl::Variable>& input);

  fl::Variable applyPosEmb(const fl::Variable& input, const int offset) const;

  std::vector<fl::Variable> vectorizedDecoder(
      const fl::Variable& encoded,
      const fl::Variable& target);


  std::pair<fl::Variable, EDState> decodeStep( 
      const fl::Variable& encoded,
      const fl::Variable& y,
      const EDState& inState) const;


  std::vector<CandidateHypo> beamSearch(
      const af::array& input,
      std::vector<EncDecCriterion::CandidateHypo> beam,
      int beamSize,
      int maxLen,
      float eos_score);

  std::vector<int> beamPath(const af::array& input, int beamSize = 10, float eos_score = 0);
   std::vector<CandidateHypo> beamSearchRes(const af::array& input, int beamSize = 10, float eos_score = 0);


/*
  std::pair<std::vector<std::vector<float>>, std::vector<EDStatePtr>>
  decodeBatchStep(
      const fl::Variable& xEncoded,
      std::vector<fl::Variable>& ys,
      const std::vector<EDState*>& inStates,
      //const int attentionThreshold,
      const float smoothingTemperature) const;
*/


  std::string prettyString() const override;

  std::shared_ptr<fl::Embedding> embedding() const {
    return std::static_pointer_cast<fl::Embedding>(module(0));
  }


  std::shared_ptr<w2l::TransformerBlockSimple> layerEnc(int i) const {
    return std::static_pointer_cast<w2l::TransformerBlockSimple>(module(i + 1));
  }

  std::shared_ptr<w2l::TransformerBlockAttend> layerDec(int i) const {
    return std::static_pointer_cast<w2l::TransformerBlockAttend>(module(i + nLayerEnc_ + 1));
  }

  
  std::shared_ptr<fl::Linear> linearOut() const {
    return std::static_pointer_cast<fl::Linear>(module(nLayerEnc_ + nLayerDec_ + 1));
  }
  

  fl::Variable startEmbedding() const {
    return params_.back();
  }

 private:
  int nClass_;
  int eos_;
  int maxDecoderOutputLen_;
  int nLayerEnc_;
  int nLayerDec_;
  bool useSinPosEmb_;
  bool posEmbEveryLayer_;
  //std::shared_ptr<WindowBase> window_;
  //bool trainWithWindow_;
  double labelSmooth_;
  double pctTeacherForcing_;
  fl::Variable sinPosEmb;

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      nClass_,
      eos_,
      maxDecoderOutputLen_,
      nLayerEnc_,
      nLayerDec_,
      //window_,
      //trainWithWindow_,
      labelSmooth_,
      pctTeacherForcing_,
      useSinPosEmb_,
      posEmbEveryLayer_)

  EncDecCriterion() = default;
};

/*
EncDecCriterion buildEncDecCriterion(
    int numClasses,
    int numLayers,
    float dropout,
    float layerdrop,
    int eosIdx);
*/

/*
struct TS2SDecoderBuffer {
  fl::Variable input;
  TS2SState dummyState;
  std::vector<fl::Variable> ys;
  std::vector<TS2SState*> prevStates;
  int attentionThreshold;
  double smoothingTemperature;

  TS2SDecoderBuffer(int beamSize, int attnThre, float smootTemp)
      : attentionThreshold(attnThre), smoothingTemperature(smootTemp) {
    ys.reserve(beamSize);
    prevStates.reserve(beamSize);
  }
};

AMUpdateFunc buildTransformerAmUpdateFunction(
    std::shared_ptr<SequenceCriterion>& crit);
*/
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::EncDecCriterion)
