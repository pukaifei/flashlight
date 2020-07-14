/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <math.h>

#include "experimental/lead2Gold/src/common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "experimental/lead2Gold/src/data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include <iostream>
#include <fstream>

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " [data_path] [dataset_name] [flags]");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  

  if (0) {
    std::vector<fl::Variable> noiselmparams;
    std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
    std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
    W2lSerializer::load(FLAGS_am, cfg, network, criterion, netoptim, critoptim, noiselmparams);
  } else{
    W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  }


  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */

  //auto tokenDict = createTokenDict();
  Dictionary tokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  //auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  //auto wordDict = createWordDict(lexicon);
  //LOG(INFO) << "Number of words: " << wordDict.indexSize();

  //DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}, {kNoiseKeyIdx, tokenDict},{kCleanKeyIdx, tokenDict}};
  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kNoiseKeyIdx, tokenDict}, {kCleanKeyIdx, tokenDict}};

  Dictionary wordDict;
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
    dicts.insert({kWordIdx, wordDict});
  }

  /* ===================== Create Dataset ===================== */
  int worldRank = 0;
  int worldSize = 1;
  auto ds = createDataset(
      FLAGS_test, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  int nSamples = ds->size();
  ds->shuffle(FLAGS_seed);

  LOG(INFO) << "[Dataset] Dataset loaded.";

  /*  Construct ForceAlignBeamNoise criterion  */
  
  int N_=29;
  w2l::Dictionary noise_keys;
  std::string token_list = "|'abcdefghijklmnopqrstuvwxyz";
  for(int i = 0; i < N_-1; i++) {
    std::string s(1, token_list[i]);
    noise_keys.addEntry(s,i);
  }

  std::shared_ptr<NoiseTrie> noiselex = nullptr;
  if (FLAGS_uselexicon && !FLAGS_lexicon.empty()) {
    noiselex = std::shared_ptr<NoiseTrie>(new NoiseTrie(tokenDict.indexSize() - FLAGS_replabel, tokenDict.getIndex("|"), nullptr));
    auto words = noiselex->load(FLAGS_lexicon, tokenDict);
  }



  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
                                                    FLAGS_noiselmtype, noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
                                                    false, FLAGS_scale_noise, FLAGS_scale_sub, FLAGS_scale_ins, FLAGS_scale_del, FLAGS_tkn_score);


  auto fac_beam = ForceAlignBeamNoise(tokenDict, noiselex, *noiselm, FLAGS_beamsize, FLAGS_beamthreshold, FLAGS_topk, true, FLAGS_useNoiseToSort);
  auto fac_beam_no_noise = ForceAlignBeamNoise(tokenDict, noiselex, *noiselm, FLAGS_beamsize, FLAGS_beamthreshold, FLAGS_topk, false, false);
  auto fac_asg = ForceAlignmentCriterion(N_, w2l::CriterionScaleMode::NONE);

  int sample_id = 0;
  auto mtr_wLER = fl::AverageValueMeter();
  auto mtr_wLER_pad = fl::AverageValueMeter();
  auto statsbeamsize = FLAGS_statsbeamsize;
  fl::TimeMeter meter_forward, meter_wLER;
  fl::AverageValueMeter meter_scoreBeam;
  fl::EditDistanceMeter mtr_LER_truth;
  for (auto& sample : *ds) {
    sample_id += 1;
    auto input = fl::input(sample[kInputIdx]);
    //for (const auto& p : network->params()) {
    //  af::print("param", p.array());
    //}
    fl::Variable emission_eval = fl::Variable();
    if (FLAGS_useevalemission){
      network->eval();
      emission_eval = network->forward({fl::input(sample[kInputIdx])}).front();
      network->train();
    }

    network->train();
    auto emission_train = network->forward({input}).front();
    int T = emission_train.dims(1);
    int B = emission_train.dims(2);
    auto transition = criterion->param(0);
    auto target = fl::noGrad(sample[kTargetIdx]);
    auto ktarget = fl::noGrad(sample[kNoiseKeyIdx]);
    int L = ktarget.dims(0);
    auto ktargetClean = fl::noGrad(sample[kCleanKeyIdx]);

    for (int b=0 ; b < B; b++){
      mtr_LER_truth.add(sample[kTargetIdx](af::span,b), sample[kCleanKeyIdx](af::span,b));
    }
    //emission = fl::logSoftmax(emission, 0);
    //transition = 0 * transition;
    bool compute_no_noise=true;

    fl::Variable fac_beam_output, fac_beam_output_no_noise;
    meter_forward.resume();
    if (FLAGS_useevalemission){
      fac_beam_output = fac_beam.forward(emission_train, emission_eval, transition, target, ktarget);
      
      //fac_beam_output = fac_beam.forward(emission_eval, emission_eval, transition, target, ktarget);
      //fac_beam_output = fac_beam.forward(emission_eval, emission_train, transition, target, ktarget);
      if (compute_no_noise){
        fac_beam_output_no_noise = fac_beam_no_noise.forward(emission_train, emission_eval, transition, target, ktarget);
      }
    } else{
      fac_beam_output = fac_beam.forward(emission_train, transition, target, ktarget);
      if (compute_no_noise){
        fac_beam_output_no_noise = fac_beam_no_noise.forward(emission_train, transition, target, ktarget);
      }
    }
    meter_forward.stop();

    meter_scoreBeam.add(fac_beam_output.array());

    //af::print("ktargetClean", ktargetClean.array()(af::span, 0));
    //af::print("target", target.array()(af::span, 0));
    //af::print("ktarget", ktarget.array()(af::span, 0));
    //af::print("fac_beam_output", fac_beam_output.array());
    //af::print("output end", emission(af::span, af::seq(af::end - 9,af::end), 0).array());
    meter_wLER.resume();
    auto wLER_paths = fac_beam.wLER(fac_beam_output, ktargetClean, statsbeamsize, &mtr_wLER);
    auto& wLER_b = std::get<0>(wLER_paths);
    auto& all_paths_weights = std::get<1>(wLER_paths);
    meter_wLER.stop();

    std::cout << "batch " << sample_id << ", wLER: " << wLER_b << ", T= "<< T << ", L= " << L << std::endl;
    //af::print("input end", input(af::seq(af::end - 9,af::end), af::span,0,0).array());
    //af::print("output end", emission(af::span, af::seq(af::end - 9,af::end), 0).array());

    //af::print("input start", input(af::seq(0, 9), af::span,0,0).array());
    //af::print("output start", emission(af::span, af::seq(0,9), 0).array());
    

    /* DISPLAY Results sentences */
    int display_first_n = 30;
    //auto paths_and_weights = fac_beam.extractPathsAndWeightsBackward(fac_beam_output, 0, FLAGS_statsbeamsize);
    fl::EditDistanceMeter mtr_LER;
    for (int b = 0; b < all_paths_weights.size(); b++ ){
      auto& paths_and_weights = all_paths_weights[b];
      std::cout << "clean: ";

      auto tgt_clean = ktargetClean.array()(af::span, b);
      auto tgtraw_clean = w2l::afToVector<int>(tgt_clean);
      auto tgtsz_clean = w2l::getTargetSize(tgtraw_clean.data(), tgtraw_clean.size());
      tgtraw_clean.resize(tgtsz_clean);

      auto tgt = ktarget.array()(af::span, b);
      auto tgtraw = w2l::afToVector<int>(tgt);
      auto tgtsz = w2l::getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(tgtsz);


      for (int j=0; j<tgtraw_clean.size(); j++){
        if (tgtraw_clean[j] == 28){
          std::cout << noise_keys.getEntry(tgtraw_clean[j-1]);
        } else{
          std::cout << noise_keys.getEntry(tgtraw_clean[j]);
        }
      }
      std::cout << std::endl;

      std::cout << "provided: ";
      for (int j=0; j<tgtraw.size(); j++){
        if (tgtraw[j] == 28){
          std::cout << noise_keys.getEntry(tgtraw[j-1]);
        } else{
          std::cout << noise_keys.getEntry(tgtraw[j]);
        }
      }
      std::cout << std::endl;

      for (int k =0 ; k < std::min(display_first_n, (int)paths_and_weights.size()) ; k++){
        auto& path_weight = paths_and_weights[k];
        auto& proposed_path = std::get<0>(path_weight);
        auto& score_path = std::get<1>(path_weight);
        mtr_LER.reset();
        mtr_LER.add(proposed_path.data(), tgtraw_clean.data(), proposed_path.size(), tgtraw_clean.size());
        double LER = mtr_LER.value()[0] ;
        if (compute_no_noise){
          double scoreClean_no_noise = fac_beam_no_noise.getTrueScore(fac_beam_output_no_noise, b, proposed_path);
          std::cout << "Score: " << scoreClean_no_noise;
          std::cout << " Noise only: " << score_path - scoreClean_no_noise;
        } else{
          std::cout << "Score: " << score_path;
        }
        std::cout << " LER: " << LER << "   ";
        std::cout << " ";
        for (int j=0; j<proposed_path.size(); j++){
          if (proposed_path[j] == 28){
            std::cout << noise_keys.getEntry(proposed_path[j-1]);
          } else{
            std::cout << noise_keys.getEntry(proposed_path[j]);
          }
        }
        std::cout << std::endl;
      }

      double scoreClean = fac_beam.getTrueScore(fac_beam_output, b, tgtraw_clean);

      double scoreProvided = fac_beam.getTrueScore(fac_beam_output, b, tgtraw);
      
      if (compute_no_noise){
        double scoreClean_no_noise = fac_beam_no_noise.getTrueScore(fac_beam_output_no_noise, b, tgtraw_clean);
        double scoreProvided_no_noise = fac_beam_no_noise.getTrueScore(fac_beam_output_no_noise, b, tgtraw);

        std::cout << "Score clean: " << scoreClean << std::endl;
        std::cout << "Score clean no noise: " << scoreClean_no_noise << std::endl;
        std::cout << "Score noise clean: " << scoreClean - scoreClean_no_noise << std::endl;
        
        auto resCleanASG = fac_asg.forward(emission_train(af::span, af::span, b), ktargetClean(af::span,b));
        float scoreCleanASG = resCleanASG.scalar<float>();
        std::cout << "Score clean ASG: " << scoreCleanASG << std::endl;

        std::cout << "Score Provided: " << scoreProvided << std::endl;
        std::cout << "Score Provided no noise: " << scoreProvided_no_noise << std::endl;
        std::cout << "Score noise Provided: " << scoreProvided - scoreProvided_no_noise << std::endl;

        auto resProvidedASG = fac_asg.forward(emission_train(af::span, af::span, b), ktarget(af::span,b));
        float scoreProvidedASG = resProvidedASG.scalar<float>();
        std::cout << "Score Provided ASG: " << scoreProvidedASG << std::endl;

      } else{
        std::cout << "Score clean: " << scoreClean << std::endl;
        std::cout << "Score Provided: " << scoreProvided << std::endl;
      }
      std::cout << std::endl;
    }


    /* STORE  */
    /*
    af::array target_af = sample[kTargetIdx](af::span, 0);
    af::array ktarget_af = sample[kNoiseKeyIdx](af::span, 0);
    af::array clean_af = sample[kCleanKeyIdx](af::span, 0);
    W2lSerializer::save("/checkpoint/adufraux/saveExample/compare_batchsize/bs_80.bin",
      emission(af::span, af::span, 0).array(),
      transition.array(),
      target_af,
      ktarget_af,
      clean_af
    );
    */

    /* TEST ADD SOME PADDING */
    /*
    int nb_pad = 10;
    std::cout << nb_pad << " pad token are added" << std::endl;
    auto input_padded = fl::input(sample[kInputIdx])
    emission = network->forward({input_padded}).front();
    T = emission.dims(1);

    fac_beam_output = fac_beam.forward(emission, transition, target, ktarget);
    double wLER_b = fac_beam.wLER(fac_beam_output, targetclean, statsbeamsize, &mtr_wLER_pad);
    
    std::cout << "batch " << sample_id << ", wLER: " << wLER_b << ", T= "<< T << ", L= " << L << std::endl;
    */
   



  }

  if (!FLAGS_computeStatsStorePath.empty()){
    std::ofstream outfile (FLAGS_computeStatsStorePath);
    outfile << std::to_string(mtr_wLER.value()[0]);
    outfile << "\n";
    outfile << std::to_string(mtr_wLER.value()[1]);
    outfile.close();
  }

  std::cout << "wLER computed with " << mtr_wLER.value()[2] << " samples" << std::endl;
  std::cout << "wLER mean: " << mtr_wLER.value()[0] << std::endl;
  std::cout << "wLER var: " << mtr_wLER.value()[1] << ", wLER standard deviation: "<< sqrt(mtr_wLER.value()[1]) << std::endl;
  std::cout << "Score beam mean: " << meter_scoreBeam.value()[0] << std::endl;
  std::cout << "Forward done in: " << meter_forward.value() << std::endl;
  std::cout << "wLER done in: " << meter_wLER.value() << std::endl;
  std::cout << "baseline LER clean/provided: " << mtr_LER_truth.value()[0] << ", standard deviation: " << sqrt(mtr_LER_truth.value()[1]) << std::endl;
  //std::cout << "pad wLER computed with " << mtr_wLER_pad.value()[2] << " samples" << std::endl;
  //std::cout << "pad wLER mean: " << mtr_wLER_pad.value()[0] << std::endl;
  //std::cout << "pad wLER var: " << mtr_wLER_pad.value()[1] << ", wLER standard deviation: "<< sqrt(mtr_wLER.value()[1]) << std::endl;

  return 0;
}
