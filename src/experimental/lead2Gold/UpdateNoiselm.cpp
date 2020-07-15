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
#include "runtime/SpeechStatMeter.h"
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

  /* ================ Set up distributed environment ================ */
  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(
        1.0 / fl::getWorldSize(), true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

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

  auto ds = createDataset(FLAGS_train, dicts, lexicon, 1, worldRank, worldSize);
  int nSamples = ds->size();
  ds->shuffle(FLAGS_seed);

  auto ds_valid =
      createDataset(FLAGS_valid, dicts, lexicon, 1, worldRank, worldSize);

  LOG(INFO) << "[Dataset] Dataset loaded.";

  /*  Construct ForceAlignBeamNoise criterion  */

  int N_ = 29;
  w2l::Dictionary noise_keys;
  std::string token_list = "|'abcdefghijklmnopqrstuvwxyz";
  for (int i = 0; i < N_ - 1; i++) {
    std::string s(1, token_list[i]);
    noise_keys.addEntry(s, i);
  }

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm =
      std::make_shared<NoiseLMLetterSwapUnit>(
          "",
          "zeronoiselm",
          noise_keys,
          FLAGS_allowSwap,
          FLAGS_allowInsertion,
          FLAGS_allowDeletion,
          false,
          FLAGS_scale_noise,
          1,
          1,
          1,
          0);

  if (isMaster)
    noiselm->displayNoiseModel();

  w2l::SpeechStatMeter speechmtr;
  noiselm->trainModel(
      ds,
      network,
      criterion,
      dicts,
      FLAGS_enable_distributed,
      (int)FLAGS_replabel,
      speechmtr);

  if (isMaster)
    noiselm->displayNoiseModel();
  // noiselm->displayNoiseModel(true);

  fl::Variable editDist(
      af::array & str1_af,
      af::array & str2_af,
      w2l::DictionaryMap & dicts,
      int replabel);

  auto p_noise_clean = fl::AverageValueMeter();
  int sample_id = 0;
  int nbRemoved = 0;
  for (auto& sample : *ds_valid) {
    sample_id++;
    network->eval();
    auto output = network->forward({fl::input(sample[kInputIdx])}).front();
    auto updatedTranscripts = getUpdateTrancripts(output, criterion, dicts);

    af::array cleanTarget = sample[w2l::kTargetIdx];
    af::array noisyTarget = updatedTranscripts[1];

    noiselm->evalBatch(
        cleanTarget, noisyTarget, dicts, FLAGS_replabel, p_noise_clean);
  }

  std::cout << "p_noise_clean: " << p_noise_clean.value()[0]
            << ", standard deviation: " << sqrt(p_noise_clean.value()[1])
            << std::endl;
  std::cout << "Removed: " << nbRemoved << std::endl;
  return 0;
}
