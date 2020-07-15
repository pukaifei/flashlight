#include "experimental/lead2Gold/src/criterion/l2g/CtcBeamNoiseCriterion.h"
#include <math.h>

// removed LM& lm,
CtcBeamNoiseCriterion::CtcBeamNoiseCriterion(
    int N,
    w2l::Dictionary& tokenDict,
    std::shared_ptr<NoiseTrie> lex,
    NoiseLMLetterSwapUnit& noiselm,
    long beamsize,
    w2l::CriterionScaleMode scaleMode,
    double threshold,
    bool computeStats,
    int top_k,
    bool useevalemission,
    bool useNoiseToSort,
    int Nb_nested)
    : fal_(
          tokenDict,
          lex,
          noiselm,
          beamsize,
          threshold,
          top_k,
          true,
          useNoiseToSort,
          Nb_nested),
      scaleMode_(scaleMode),
      N_(N),
      computeStats_(computeStats),
      useevalemission_(useevalemission),
      Nb_nested_(Nb_nested) {}

std::vector<fl::Variable> CtcBeamNoiseCriterion::forward(
    fl::Variable& emissions,
    fl::Variable& emissions_eval,
    fl::Variable target,
    fl::Variable cleantarget,
    int statsbeamsize,
    fl::AverageValueMeter* mtr_wLER) {
  if (af::anyTrue<bool>(af::isNaN(emissions.array()))) {
    throw std::invalid_argument("emissions has nan");
  }

  int N = emissions.dims(0);
  int T = emissions.dims(1);
  int B = emissions.dims(2);
  int mS = target.dims(0);

  // compute scaling factor
  std::vector<int> targetVec(target.elements());
  target.host(targetVec.data());
  std::vector<int> targetSizeVec(B);

  w2l::cpu::CriterionUtils<float>::batchTargetSize(
      B, mS, mS, targetVec.data(), targetSizeVec.data());

  std::vector<float> scaleVec(B);
  w2l::cpu::CriterionUtils<float>::computeScale(
      B, T, N, scaleMode_, targetSizeVec.data(), scaleVec.data());
  fl::Variable scale(af::array(B, scaleVec.data()), false);

  fl::Variable fal_output_beam;
  if (useevalemission_) {
    fal_output_beam = fal_.forward(emissions, emissions_eval, target);
  } else {
    fal_output_beam = fal_.forward(emissions, target);
  }

  // if (computeStats_){
  // auto wLER_paths = fal_.wLER(fal_output_beam, cleantarget, statsbeamsize,
  // mtr_wLER);
  //}

  // auto fal_output_beam_scaled = scale * fal_output_beam;

  return {-1 * scale * fal_output_beam};
}

/*
af::array AutoSegBeamNoiseCriterion::viterbi(const fl::Variable& output)
{
  return fal_.viterbi(output);
}
*/
