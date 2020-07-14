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

std::vector<double> editDist(std::vector<int> str1, std::vector<int> str2, bool noise_objectif = false, std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = nullptr) 
{ 
    //str1 clean to str2 noisy
    // Create a table to store results of subproblems 
    int m = str1.size();
    int n = str2.size();
    double dp[m + 1][n + 1];
    double dp_noise[m + 1][n + 1];

    int decision;

    double dp_noise_ij_ins, dp_noise_ij_del, dp_noise_ij_sub, dp_ij_ins, dp_ij_del, dp_ij_sub;
  
    // Fill d[][] in bottom up manner 
    for (int i = 0; i <= m; i++) { 
        for (int j = 0; j <= n; j++) { 
            // If first string is empty, only option is to 
            // insert all characters of second string
            if (i == 0 && j==0){
              dp[i][j] = 0;
              if (noiselm)
                dp_noise[i][j] = 0;
            }

            else if (i == 0){ // ---> insertion first line
              dp[i][j] = j; // Min. operations = j 
              if (noiselm)
                dp_noise[i][j] = dp_noise[i][j-1] + noiselm->scoreInsertion(str2[j-1]);       
            }
  
            // If second string is empty, only option is to 
            // remove all characters of second string 
            else if (j == 0){
              dp[i][j] = i; // Min. operations = i
              if (noiselm)
                dp_noise[i][j] = dp_noise[i-1][j] + noiselm->scoreDeletion(str1[i-1]);
            }

            else if (noiselm){

              if (str1[i - 1] == str2[j - 1])
                  dp_ij_sub = dp[i - 1][j - 1];
              else
                dp_ij_sub = 1 + dp[i - 1][j - 1];
              dp_ij_ins = 1 + dp[i][j - 1];
              dp_ij_del = 1 + dp[i - 1][j];

              dp_noise_ij_ins = dp_noise[i][j - 1] + noiselm->scoreInsertion(str2[j-1]);
              dp_noise_ij_del = dp_noise[i-1][j] + noiselm->scoreDeletion(str1[i-1]);
              dp_noise_ij_sub = dp_noise[i-1][j-1] + noiselm->scoreSwap(str2[j-1], str1[i-1]);

              if (noise_objectif){
                if (dp_noise_ij_sub > dp_noise_ij_del && dp_noise_ij_sub > dp_noise_ij_ins){
                  dp_noise[i][j] = dp_noise_ij_sub;
                  dp[i][j] = dp_ij_sub;
                } else{
                  if (dp_noise_ij_del > dp_noise_ij_ins){
                    dp_noise[i][j] = dp_noise_ij_del;
                    dp[i][j] = dp_ij_del;
                  } else{
                    dp_noise[i][j] = dp_noise_ij_ins;
                    dp[i][j] = dp_ij_ins;
                  }
                }
              } else{
                if (dp_ij_sub < dp_ij_del && dp_ij_sub < dp_ij_ins){
                  dp_noise[i][j] = dp_noise_ij_sub;
                  dp[i][j] = dp_ij_sub;
                } else{
                  if (dp_ij_del < dp_ij_ins){
                    dp_noise[i][j] = dp_noise_ij_del;
                    dp[i][j] = dp_ij_del;
                  } else{
                    dp_noise[i][j] = dp_noise_ij_ins;
                    dp[i][j] = dp_ij_ins;
                  }
                }
              }            
            } else{
              // If last characters are same, ignore last char 
              // and recur for remaining string 
              if (str1[i - 1] == str2[j - 1])
                  dp[i][j] = dp[i - 1][j - 1]; 
    
              // If the last character is different, consider all 
              // possibilities and find the minimum 
              else
                  dp[i][j] = 1 + std::min(dp[i][j - 1], // Insert 
                                     std::min(dp[i - 1][j], // Remove 
                                                dp[i - 1][j - 1])); // Replace 
            }
        } 
    } 
  
    if (noiselm)
      return {dp[m][n], dp_noise[m][n]};
    else
      return {dp[m][n]}; 
}


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


  //std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
  //                                                  FLAGS_noiselmtype, noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
  //                                                  false, FLAGS_scale_noise, FLAGS_scale_sub, FLAGS_scale_ins, FLAGS_scale_del, FLAGS_tkn_score);

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm_raw = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
                                                    FLAGS_noiselmtype, noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
                                                    false, 1, 1, 1, 1, 0);



  //auto fac_beam = ForceAlignBeamNoise(tokenDict, noiselex, *noiselm, FLAGS_beamsize, FLAGS_beamthreshold, FLAGS_topk, true, FLAGS_useNoiseToSort);
  auto fac_asg = ForceAlignmentCriterion(N_, w2l::CriterionScaleMode::NONE);

  int sample_id = 0;
  auto mtr_alpha = fl::AverageValueMeter();

  network->train();
  for (auto& sample : *ds) {
    sample_id += 1;
    auto input = fl::input(sample[kInputIdx]);


    auto emission_train = network->forward({input}).front();

    int T = emission_train.dims(1);
    int B = emission_train.dims(2);
    auto transition = criterion->param(0);
    auto target = fl::noGrad(sample[kTargetIdx]);
    auto ktarget = fl::noGrad(sample[kNoiseKeyIdx]);
    int L = ktarget.dims(0);
    auto ktargetClean = fl::noGrad(sample[kCleanKeyIdx]);

    auto fac_noise = fac_asg.forward(emission_train, ktarget);
    auto fac_noise_v = w2l::afToVector<float>(fac_noise);

    auto fac_clean = fac_asg.forward(emission_train, ktargetClean);
    auto fac_clean_v = w2l::afToVector<float>(fac_clean);

//#pragma omp parallel for num_threads(B)
    for (int b=0; b < B; b++){
      auto tgt_clean = ktargetClean.array()(af::span, b);
      auto tgtraw_clean = w2l::afToVector<int>(tgt_clean);
      auto tgtsz_clean = w2l::getTargetSize(tgtraw_clean.data(), tgtraw_clean.size());
      tgtraw_clean.resize(tgtsz_clean);

      auto tgt_noisy = ktarget.array()(af::span, b);
      auto tgtraw_noisy = w2l::afToVector<int>(tgt_noisy);
      auto tgtsz_noisy = w2l::getTargetSize(tgtraw_noisy.data(), tgtraw_noisy.size());
      tgtraw_noisy.resize(tgtsz_noisy);

      double alpha;
      if (tgtraw_clean == tgtraw_noisy){
        alpha = 0;
      } else{
        auto p_noise_clean = editDist(tgtraw_clean, tgtraw_noisy, true, noiselm_raw)[1];
        auto p_clean_clean = editDist(tgtraw_clean, tgtraw_clean, true, noiselm_raw)[1];
        alpha = (fac_noise_v[b] - fac_clean_v[b] ) / (p_clean_clean - p_noise_clean);
        std::cout << "fac_noise " << fac_noise_v[b] << " fac_clean " << fac_clean_v[b] << std::endl;
        std::cout << "p_clean_clean " << p_clean_clean << " p_noise_clean " << p_noise_clean << std::endl;
      }
      std::cout << "alpha: " << alpha << std::endl;

      mtr_alpha.add(alpha);

      //auto edit_dist_noiseobj = editDist(tgtraw_clean, tgtraw_noisy, true, noiselm_raw);
      //auto edit_dist = editDist(tgtraw_clean, tgtraw_noisy, false, noiselm_raw);

      //std::cout << "OBJ Noise, edit: " << edit_dist_noiseobj[0] << " noise: " << edit_dist_noiseobj[1] << std::endl;
      //std::cout << "     Edit, edit: " << edit_dist[0] << " noise: " << edit_dist[1] << std::endl;
    }
    std::cout << "AVERAGE ALPHA " << mtr_alpha.value()[0] << " sd: " << sqrt(mtr_alpha.value()[1]) << std::endl;

    
  }
/*
  if (!FLAGS_computeStatsStorePath.empty()){
    std::ofstream outfile (FLAGS_computeStatsStorePath);
    outfile << std::to_string(mtr_clean.value()[0]);
    outfile << "\n";
    outfile << std::to_string(mtr_noisy.value()[0]);
    outfile << "\n";
    outfile << std::to_string(mtr_clean.value()[0] - mtr_noisy.value()[0]);
    outfile.close();
  }
  std::cout << "Computed with " << mtr_clean.value()[2] << " samples" << std::endl;
  std::cout << "L2G clean: " << mtr_clean.value()[0] << ", standard deviation: " << sqrt(mtr_clean.value()[1]) << std::endl;
  std::cout << "L2G noisy: " << mtr_noisy.value()[0] << ", standard deviation: " << sqrt(mtr_noisy.value()[1]) << std::endl;
  std::cout << "L2G clean - noisy: " << mtr_clean.value()[0] - mtr_noisy.value()[0] << std::endl;
*/
  return 0;
}
