#include "experimental/lead2Gold/src/criterion/l2g/AutoSegBeamNoiseCriterion.h"
#include <math.h>

// removed LM& lm,
AutoSegBeamNoiseCriterion::AutoSegBeamNoiseCriterion(
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
    bool useNoiseToSort)
    : fal_(
          tokenDict,
          lex,
          noiselm,
          beamsize,
          threshold,
          top_k,
          true,
          useNoiseToSort),
      falStats_(tokenDict, lex, noiselm, beamsize, threshold),
      fcc_(N, scaleMode),
      scaleMode_(scaleMode),
      N_(N),
      fal_true_(N, scaleMode),
      computeStats_(computeStats),
      useevalemission_(useevalemission) {
  // scaleFn_ = getCriterionScaleFn(scaleMode_);
  // std::cout << "Valeur de B : " << B << std::endl;
  // std::cout << "Valeur de N : " << N << std::endl;
}

std::vector<fl::Variable> AutoSegBeamNoiseCriterion::forward(
    fl::Variable& emissions,
    fl::Variable& emissions_eval,
    fl::Variable transitions,
    fl::Variable target,
    fl::Variable ktarget,
    fl::Variable cleantarget,
    int statsbeamsize,
    fl::AverageValueMeter* mtr_wLER) {
  if (af::anyTrue<bool>(af::isNaN(emissions.array()))) {
    throw std::invalid_argument("emissions has nan");
  }
  if (af::anyTrue<bool>(af::isNaN(transitions.array()))) {
    throw std::invalid_argument("transitions has nan");
  }

  int N = emissions.dims(0);
  int T = emissions.dims(1);
  int B = emissions.dims(2);
  int mS = target.dims(0);

  /*
  // print things

  std::cout << "" << std::endl;
  af::print("emissions", emissions(af::span,af::seq(0,10)).array());
  for(int b = 0; b < B ; b++){
    for(int t = 0; t < T ; t++){
      for(int n = 0; n < N ; n++){
        std::cout << std::to_string(emissions(n,t,b).scalar<float>()) << ",";
      }
    }
  }

  std::cout << std::endl << std::endl << std::endl;
  std::cout << "target3" << std::endl;
  for(int b = 0; b < B ; b++){
    std::cout << "B= " << b << std::endl;
    for(int l = 0; l < mS ; l++){
      std::cout << "l= " << l << " ";
      std::cout << std::to_string(target(l,b).scalar<int>()) << ",";
    }
  }

  int64_t mkS = ktarget.dims(0);
  std::cout << "ktarget" << std::endl;
  for(int b = 0; b < B ; b++){
    for(int l = 0; l < mkS ; l++){
      std::cout << std::to_string(ktarget(l,b).scalar<int>()) << ",";
    }
  }
  std::cout << std::endl << std::endl << std::endl;
  af::print("transition", transitions(af::seq(0,10),af::seq(0,10)).array());
  for(int n1 = 0; n1 < N ; n1++){
    for(int n2 = 0; n2 < N ; n2++){
      std::cout << std::to_string(transitions(n2,n1).scalar<float>()) << ",";
    }
  }
  std::cout << std::endl << std::endl << std::endl;
  */

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

  // already scaled
  fcc_.setParams(transitions, 0);
  // target is only use to scale.
  // af::print("emission train", emissions.array());
  // af::print("emission eval", emissions_eval.array());
  auto fcc_output = fcc_.forward(emissions, target);

  fl::Variable fal_output_beam;
  if (useevalemission_) {
    // emissions.array() = emissions_eval.array();
    fal_output_beam =
        fal_.forward(emissions, emissions_eval, transitions, target, ktarget);
  } else {
    fal_output_beam = fal_.forward(emissions, transitions, target, ktarget);
  }
  // af::print("new emission train", emissions.array());

  // af::print("fcc_output", fcc_output.array());
  // af::print("fal_output_beam", fal_output_beam.array());
  // af::print("scale", scale.array());
  // auto scaled = scale * fal_output_beam;
  // af::print("fal_output_beam scaled", scaled.array());

  // auto fal_output_beam = fal_.forward(emissions, transitions, target,
  // ktarget); af::print("fal_output_beam emissions_train",
  // fal_output_beam.array());

  if (computeStats_) {
    auto wLER_paths =
        fal_.wLER(fal_output_beam, cleantarget, statsbeamsize, mtr_wLER);
  }

  auto fal_output_beam_scaled = scale * fal_output_beam;
  auto tot_loss = fcc_output - fal_output_beam_scaled;

  auto nanValues = af::isNaN(tot_loss.array());
  if (af::anyTrue<bool>(
          nanValues)) { // C'est du bricolage, ça n'arrive quasiment jamais
    // std::vector<int> nanValues_v(score.elements());
    // score.host(score_v.data());
    af::print("nanValues ", nanValues);
    for (int64_t b = 0; b < B; b++) {
      // std::cout << "score " << b << " " << nanValues(b) << std::endl;
      if (af::anyTrue<bool>(nanValues(b))) {
        std::cout << "CHANGE !" << std::endl;
        tot_loss.array()(b) = 0.0;
        fcc_output.array()(b) = 0.0;
        fal_output_beam.array()(b) = 0.0;
      }
    }
    // score = score * (1 - fl::Variable(nanValues,false));
    af::print("fcc_output", fcc_output.array());
    af::print("fal_output_beam", fal_output_beam.array());
    af::print("scale", scale.array());
    std::cout << "N " << N << " T " << T << " B " << B << std::endl;
    af::print("score", tot_loss.array());
  }

  // af::print("score", score.array());
  // score.setPayload(fal_output_beam.getPayload());
  return {tot_loss, fcc_output, fal_output_beam_scaled};
}

// fl::Variable AutoSegBeamNoiseCriterion::forward(fl::Variable emissions,
// fl::Variable transitions)
//{
//  return fal_.forward(emissions, transitions);
//}

af::array AutoSegBeamNoiseCriterion::viterbi(const fl::Variable& output) {
  return fal_.viterbi(output);
}
/*
af::array AutoSegBeamNoiseCriterion::viterbiWord(const fl::Variable& output)
{
  return fal_.viterbiWord(output);
}
*/
