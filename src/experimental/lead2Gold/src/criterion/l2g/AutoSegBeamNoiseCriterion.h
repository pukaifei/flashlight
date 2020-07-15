#pragma once

#include "criterion/CriterionUtils.h"
#include "criterion/FullConnectionCriterion.h"
#include "experimental/lead2Gold/src/criterion/l2g/ForceAlignBeamNoise.h"
#include "experimental/lead2Gold/src/criterion/l2g/ForceAlignBeamNoiseStats.h"
#include "libraries/common/Dictionary.h"
#include "libraries/criterion/cpu/CriterionUtils.h"

// testing purpose
#include "criterion/ForceAlignmentCriterion.h"

class AutoSegBeamNoiseCriterion {
 private:
  ForceAlignBeamNoise fal_;
  ForceAlignBeamNoiseStats falStats_;
  w2l::FullConnectionCriterion fcc_;
  w2l::CriterionScaleMode scaleMode_;
  int N_;

  // testing purpose
  w2l::ForceAlignmentCriterion fal_true_;
  bool computeStats_, useevalemission_;

 public:
  AutoSegBeamNoiseCriterion(
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
      bool useNoiseToSort = false);
  std::vector<fl::Variable> forward(
      fl::Variable& emissions,
      fl::Variable& emissions_eval,
      fl::Variable transitions,
      fl::Variable target,
      fl::Variable wtarget,
      fl::Variable cleantarget = fl::Variable(),
      int statsbeamsize = 5,
      fl::AverageValueMeter* mtr_wLER = nullptr);
  // fl::Variable forward(fl::Variable emissions, fl::Variable transitions);
  af::array viterbi(const fl::Variable& output);
  // af::array viterbiWord(const fl::Variable& output);
};
