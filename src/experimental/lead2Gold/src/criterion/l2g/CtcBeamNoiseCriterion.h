#pragma once

#include "criterion/CriterionUtils.h"
#include "experimental/lead2Gold/src/criterion/l2g/CtcForceAlignBeamNoise.h"
#include "libraries/common/Dictionary.h"
#include "libraries/criterion/cpu/CriterionUtils.h"

class CtcBeamNoiseCriterion {
 private:
  CtcForceAlignBeamNoise fal_;
  w2l::CriterionScaleMode scaleMode_;
  int N_;

  bool computeStats_, useevalemission_;
  int Nb_nested_;

 public:
  CtcBeamNoiseCriterion(
      int N,
      w2l::Dictionary& tokenDict,
      std::shared_ptr<NoiseTrie> lex,
      NoiseLMLetterSwapUnit& lm,
      long B,
      w2l::CriterionScaleMode scaleMode_,
      double threshold = 0,
      bool computeStats = false,
      int top_k = 0,
      bool useevalemission = false,
      bool useNoiseToSort = false,
      int Nb_nested = 1);
  std::vector<fl::Variable> forward(
      fl::Variable& emissions,
      fl::Variable& emissions_eval,
      fl::Variable target,
      fl::Variable cleantarget = fl::Variable(),
      int statsbeamsize = 5,
      fl::AverageValueMeter* mtr_wLER = nullptr);
  // af::array viterbi(const fl::Variable& output);
};
