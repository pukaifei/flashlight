#pragma once

#include "experimental/lead2Gold/src/criterion/l2g/ForceAlignBeamNoise.h"
//#include "criterion/FullConnectionCriterion.h"
#include "criterion/CriterionUtils.h"

// testing purpose
//#include "criterion/ForceAlignmentCriterion.h"

class AnalyseBeam {
 private:
  ForceAlignBeamNoise fal_;
  fl::EditDistanceMeter LER;
  fl::EditDistanceMeter LER_baseline;
  w2l::Dictionary& dict_;
  void backward();

 public:
  AnalyseBeam(
      NoiseTrie& keytrie,
      NoiseLMLetterSwapUnit& lm,
      long B,
      w2l::Dictionary& dict,
      double threshold = 0);
  std::vector<fl::Variable> forward(
      fl::Variable emissions,
      fl::Variable transitions,
      fl::Variable noisy_target,
      fl::Variable knoisy_target,
      fl::Variable kclean_target);
  // af::array viterbi(const fl::Variable& output);
  // af::array viterbiWord(const fl::Variable& output);
  fl::AverageValueMeter tot_wLER;
  fl::AverageValueMeter tot_LER_baseline;
};