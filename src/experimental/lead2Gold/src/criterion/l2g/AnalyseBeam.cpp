
#include "experimental/lead2Gold/src/criterion/l2g/AnalyseBeam.h"
#include "common/Utils.h"
#include "criterion/CriterionUtils.h"
#include "common/FlashlightUtils.h"

//removed LM& lm,
AnalyseBeam::AnalyseBeam(NoiseTrie& keytrie, NoiseLMLetterSwapUnit& noiselm, long B, w2l::Dictionary& dict, double threshold)
  : fal_(keytrie, noiselm, B, threshold), LER(), tot_wLER(), dict_(dict), LER_baseline(), tot_LER_baseline()
{
}

std::vector<fl::Variable> AnalyseBeam::forward(fl::Variable emissions, fl::Variable transitions, fl::Variable noisy_target, fl::Variable knoisy_target, fl::Variable kclean_target)
{
  if(af::anyTrue<bool>(af::isNaN(emissions.array()))) {
    throw std::invalid_argument("emissions has nan");
  }
  if(af::anyTrue<bool>(af::isNaN(transitions.array()))) {
    throw std::invalid_argument("transitions has nan");
  }

  int64_t N = emissions.dims(0);
  int64_t T = emissions.dims(1);
  int64_t B = emissions.dims(2);
  int64_t mS = noisy_target.dims(0);


  // print things
  //std::cout << "N = " << N << " T = " << T << " B = " << B << " L = " << mS << std::endl;
  //std::cout << "emissions" << std::endl;
  //for(int b = 0; b < B ; b++){
    //for(int t = 0; t < T ; t++){
      //for(int n = 0; n < N ; n++){
        //std::cout << std::to_string(emissions(n,t,b).scalar<float>()) << ",";
      //}
    //}
  //}
  //std::cout << std::endl << std::endl << std::endl;
  //std::cout << "target" << std::endl;
  //for(int b = 0; b < B ; b++){
  //  for(int l = 0; l < mS ; l++){
  //    std::cout << std::to_string(target(l,b).scalar<int>()) << ",";
  //  }
  //}
  //std::cout << std::endl << std::endl << std::endl;
  //std::cout << "transition" << std::endl;
  //for(int n1 = 0; n1 < N ; n1++){
  //  for(int n2 = 0; n2 < N ; n2++){
  //    std::cout << std::to_string(transitions(n2,n1).scalar<float>()) << ",";
  //  }
  //}
  //std::cout << std::endl << std::endl << std::endl;


  auto fal_output_beam = fal_.forward(emissions, transitions, noisy_target, knoisy_target);
  std::vector<float> fal_output_beam_host(B);
  fal_output_beam.host(fal_output_beam_host.data());

  //B=1;
  std::vector<std::map< std::vector<int>, double>> res_beam(B);
  //B=1;
  if (T < 150){
#pragma omp parallel for num_threads(B)
  for(int64_t b = 0; b < B; b++) { //look at each path for each example in the batch
    res_beam[b] = fal_.extractPathsAndWeights(fal_output_beam, b);
  }

std::map< std::vector<int>, double> pathsToValue;
  for(int64_t b = 0; b < B; b++) { //look at each path for each example in the batch
    pathsToValue = res_beam[b];
    std::vector<double> probas_v;
    double sum= -std::numeric_limits<double>::infinity();
    for ( const auto &p_v : pathsToValue ) {
      probas_v.push_back(p_v.second);
      sum = w2l::logSumExp(sum, p_v.second);
    } 
    auto probas = fl::softmax(fl::Variable(af::array(probas_v.size(), probas_v.data()), false), 0);
    probas.host(probas_v.data());

    auto tgt_clean = kclean_target.array()(af::span, b);
    auto tgtraw_clean = w2l::afToVector<int>(tgt_clean);
    auto tgtsz_clean = w2l::getTargetSize(tgtraw_clean.data(), tgtraw_clean.size());
    tgtraw_clean.resize(tgtsz_clean);

    auto tgt_noisy = knoisy_target.array()(af::span, b);
    auto tgtraw_noisy = w2l::afToVector<int>(tgt_noisy);
    auto tgtsz_noisy = w2l::getTargetSize(tgtraw_noisy.data(), tgtraw_noisy.size());
    tgtraw_noisy.resize(tgtsz_noisy);

    int i=0;
    double wLER=0.0;

    std::cout << " CLEAN TRANSCRIPTION" << std::endl;
    for (long j=0; j < tgtraw_clean.size(); j++){
      if (tgtraw_clean[j] == 28) {
        std::cout << dict_.getEntry(tgtraw_clean[j-1]);
      }else{
        std::cout << dict_.getEntry(tgtraw_clean[j]);
      }
    }
    std::cout << std::endl;

    LER_baseline.reset();
    LER_baseline.add(tgtraw_noisy.data(), tgtraw_clean.data(), tgtraw_noisy.size(), tgtraw_clean.size());
    tot_LER_baseline.add(LER_baseline.value()[0]);
    std::cout << "LER baseline " << LER_baseline.value()[0] << std::endl;
    std::cout << " NOISY TRANSCRIPTION" << std::endl;
    
    for (long j=0; j < tgtraw_noisy.size(); j++){
      if (tgtraw_noisy[j] == 28) {
        std::cout << dict_.getEntry(tgtraw_noisy[j-1]);
      }else{
        std::cout << dict_.getEntry(tgtraw_noisy[j]);
      }
    }
    std::cout << std::endl << "LER noisy / clean " << LER_baseline.value()[0] << std::endl;
    std::cout << "beam  " << fal_output_beam_host[b] << " sum " << sum << std::endl;

    for ( const auto &p_v : pathsToValue ) {
      LER.reset();
      std::vector<int> path = p_v.first;
      std::reverse(path.begin(), path.end()); //reverse the order of the elements
      
      
      for (long j=0; j < path.size(); j++){
        if (path[j]==28){
          path[j] = path[j-1];
        }
        std::cout << dict_.getEntry(path[j]);
      }
      LER.add(path.data(), tgtraw_clean.data(), path.size(), tgtraw_clean.size());

      if (path == tgtraw_clean){
        std::cout << " TRUTH FOUND";
      }
      std::cout << std::endl;
      std::cout << LER.value()[0] << "  " << probas_v[i] << std::endl << std::endl;

      wLER += LER.value()[0] * probas_v[i];
      i++;
    }
    tot_wLER.add(wLER);
    std::cout << "wLER " << wLER << std::endl;
  }
  }
  //std::cout << " tot wLER " << tot_wLER.value()[0] << std::endl;
  //std::cout << " tot LER baseline " << tot_LER_baseline.value()[0] << std::endl;
  
  //std::vector<int> target_v(target.elements());
  //target.host(target_v.data());
  //std::vector<float> scale_v(B);


  //auto score = 0.0 * fal_output_beam;
  auto score = fal_output_beam;
  //af::print("score", score.array());
  //score.setPayload(fal_output_beam.getPayload());
  auto grad_func = [this]() {
    this->backward();
  };

  return {score};
}

void AnalyseBeam::backward() {};

//fl::Variable AutoSegBeamNoiseCriterion::forward(fl::Variable emissions, fl::Variable transitions)
//{
//  return fal_.forward(emissions, transitions);
//}

//af::array AnalyseBeam::viterbi(const fl::Variable& output)
//{
//  return fal_.viterbi(output);
//}

//af::array AnalyseBeam::viterbiWord(const fl::Variable& output)
//{
//  return fal_.viterbiWord(output);
//}
