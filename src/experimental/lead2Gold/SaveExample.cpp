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
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
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

  LOG(INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Test ===================== */

  auto transition = criterion->param(0).array();
  int sample_id = 0;
  std::ofstream myfile (FLAGS_saveExamplePathFolder + "/list_files.txt");
  for (auto& sample : *ds) {
    sample_id += 1;
    auto emission = network->forward({fl::input(sample[kInputIdx])}).front().array();
    auto ltrTarget = sample[kTargetIdx];
    auto ltrKeyTarget = sample[kNoiseKeyIdx]; //without rep label
    auto ltrKeyTargetClean = sample[kCleanKeyIdx];
    int L = sample[kTargetIdx].dims(0);
    int N = emission.dims(0);
    int T = emission.dims(1);
    int B = emission.dims(2);

    //std::cout << "T: " << T << std::endl;
    //af::print("ltrKeyTargetClean_af", ltrKeyTargetClean);
    //af::print("ltrKeyTarget", ltrKeyTarget);
    //af::print("ltrTarget", ltrTarget);

    /* ====== Serialize emission and targets for decoding ====== */
    //std::string cleanedTestPath = cleanFilepath(FLAGS_test);
    std::string savePath = FLAGS_saveExamplePathFolder + "/" + std::to_string(sample_id) + "_size_" + std::to_string(B) + ".bin";
    myfile << savePath << "\n";
    LOG(INFO) << "[Serialization] Saving into file: " << savePath;
    W2lSerializer::save(savePath, emission, transition, ltrTarget, ltrKeyTarget, ltrKeyTargetClean);
  }
  myfile.close();

  return 0;
}
