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

#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "experimental/lead2Gold/src/common/Defines.h"
#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "experimental/lead2Gold/src/data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include <fstream>
#include <iostream>

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
    W2lSerializer::load(
        FLAGS_am, cfg, network, criterion, netoptim, critoptim, noiselmparams);
  } else {
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

  // auto tokenDict = createTokenDict();
  Dictionary tokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  // auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  // auto wordDict = createWordDict(lexicon);
  // LOG(INFO) << "Number of words: " << wordDict.indexSize();

  // DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict},
  // {kNoiseKeyIdx, tokenDict},{kCleanKeyIdx, tokenDict}};
  DictionaryMap dicts = {{kTargetIdx, tokenDict},
                         {kNoiseKeyIdx, tokenDict},
                         {kCleanKeyIdx, tokenDict}};

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

  int N_ = 29;
  w2l::Dictionary noise_keys;
  std::string token_list = "|'abcdefghijklmnopqrstuvwxyz";
  for (int i = 0; i < N_ - 1; i++) {
    std::string s(1, token_list[i]);
    noise_keys.addEntry(s, i);
  }

  std::shared_ptr<NoiseTrie> noiselex = nullptr;
  if (FLAGS_uselexicon && !FLAGS_lexicon.empty()) {
    noiselex = std::shared_ptr<NoiseTrie>(new NoiseTrie(
        tokenDict.indexSize() - FLAGS_replabel,
        tokenDict.getIndex("|"),
        nullptr));
    auto words = noiselex->load(FLAGS_lexicon, tokenDict);
  }

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm =
      std::make_shared<NoiseLMLetterSwapUnit>(
          FLAGS_probasdir,
          FLAGS_noiselmtype,
          noise_keys,
          FLAGS_allowSwap,
          FLAGS_allowInsertion,
          FLAGS_allowDeletion,
          false,
          FLAGS_scale_noise,
          FLAGS_scale_sub,
          FLAGS_scale_ins,
          FLAGS_scale_del,
          FLAGS_tkn_score);

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm_raw =
      std::make_shared<NoiseLMLetterSwapUnit>(
          FLAGS_probasdir,
          FLAGS_noiselmtype,
          noise_keys,
          FLAGS_allowSwap,
          FLAGS_allowInsertion,
          FLAGS_allowDeletion,
          false,
          1,
          1,
          1,
          1,
          1);

  auto fac_beam = ForceAlignBeamNoise(
      tokenDict,
      noiselex,
      *noiselm,
      FLAGS_beamsize,
      FLAGS_beamthreshold,
      FLAGS_topk,
      true,
      FLAGS_useNoiseToSort);
  auto fac_asg = ForceAlignmentCriterion(N_, w2l::CriterionScaleMode::NONE);
  auto transition = criterion->param(0);
  fac_asg.setParams(transition, 0);

  int sample_id = 0;
  auto mtr_clean = fl::AverageValueMeter();
  auto mtr_noisy = fl::AverageValueMeter();

  network->train();
  for (auto& sample : *ds) {
    sample_id += 1;
    auto input = fl::input(sample[kInputIdx]);

    auto emission_train = network->forward({input}).front();
    int T = emission_train.dims(1);
    int B = emission_train.dims(2);

    auto target = fl::noGrad(sample[kTargetIdx]);
    auto ktarget = fl::noGrad(sample[kNoiseKeyIdx]);
    int L = ktarget.dims(0);
    auto ktargetClean = fl::noGrad(sample[kCleanKeyIdx]);

    fl::Variable fac_asg_output_clean, fac_beam_output_noisy;
    fac_beam_output_noisy =
        fac_beam.forward(emission_train, transition, target, ktarget);
    mtr_noisy.add(fac_beam_output_noisy.array());

    fac_asg_output_clean = fac_asg.forward(emission_train, ktargetClean);
    mtr_clean.add(fac_asg_output_clean.array());

    af::print("fac_beam_output_noisy", fac_beam_output_noisy.array());
    af::print("fac_asg_output_clean", fac_asg_output_clean.array());
  }

  if (!FLAGS_computeStatsStorePath.empty()) {
    std::ofstream outfile(FLAGS_computeStatsStorePath);
    outfile << std::to_string(mtr_clean.value()[0]);
    outfile << "\n";
    outfile << std::to_string(mtr_noisy.value()[0]);
    outfile << "\n";
    outfile << std::to_string(mtr_clean.value()[0] - mtr_noisy.value()[0]);
    outfile.close();
  }
  std::cout << "Computed with " << mtr_clean.value()[2] << " samples"
            << std::endl;
  std::cout << "fal clean: " << mtr_clean.value()[0]
            << ", standard deviation: " << sqrt(mtr_clean.value()[1])
            << std::endl;
  std::cout << "L2G noisy: " << mtr_noisy.value()[0]
            << ", standard deviation: " << sqrt(mtr_noisy.value()[1])
            << std::endl;
  std::cout << "clean - noisy: " << mtr_clean.value()[0] - mtr_noisy.value()[0]
            << std::endl;

  return 0;
}
