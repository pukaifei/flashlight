/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/semisupervised/src/runtime/Init.h"

#include <tuple>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Utils.h"
#include "experimental/ConvLM/Utils.h"
#include "experimental/semisupervised/src/runtime/Defines.h"
#include "experimental/semisupervised/src/runtime/Utils.h"
#include "runtime/Serial.h"

namespace w2l {
std::unordered_map<std::string, std::string> setFlags(int argc, char** argv) {
  auto readNewFlags = [&]() {
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
  };

  auto loadOldFlags = [&](const std::string& reloadPath) {
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadPath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    auto epoch = cfg.find(kEpoch);
    LOG_IF(WARNING, epoch == cfg.end())
        << "Did not find epoch to start from, starting from 0.";

    auto startEp = epoch == cfg.end() ? 0 : std::stoi(epoch->second);
    auto startIt =
        cfg.find(kIteration) == cfg.end() ? 0 : std::stoi(cfg[kIteration]);

    return std::make_pair(startEp, startIt);
  };

  std::string runStatus = argv[1];
  std::string runPath; // current experiment path
  int runIdx = 1; // current #runs in this path
  std::string reloadPath; // path to model to reload
  int startEpoch = 0;
  int startIter = 0;

  if (runStatus == kTrainMode) {
    readNewFlags();
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    // this assumes that FLAGS_itersave wasn't set
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::tie(startEpoch, startIter) = loadOldFlags(reloadPath);
    readNewFlags();
  } else if (runStatus == kForkMode) {
    reloadPath = argv[2];
    loadOldFlags(reloadPath);
    readNewFlags();
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, argvs[0]},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      {kUserName, getEnvVar("USER")},
      {kHostName, getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath},
      // extra fields defined in semisupervised/runtime/Defines.h
      {kReloadPath, reloadPath},
      {kRunStatus, runStatus},
      {kStartEpoch, std::to_string(startEpoch)},
      {kStartIter, std::to_string(startIter)}};

  return config;
}

std::shared_ptr<fl::Module> initLM(const Dictionary& lmDict) {
  std::shared_ptr<fl::Module> lmNetwork;

  if (!FLAGS_lm.empty()) {
    W2lSerializer::load(FLAGS_lm, lmNetwork);
  } else {
    std::shared_ptr<fl::BinaryModule> lmCriterion;

    std::vector<int> adaptiveTail;
    if (FLAGS_lmcrit == kLMASCrit) {
      auto cutoffs = splitOnAnyOf(",", FLAGS_lmadasoftmaxcutoff, true);
      for (const auto& val : cutoffs) {
        adaptiveTail.push_back(std::stoi(val));
      }
      adaptiveTail.push_back(lmDict.indexSize());
    }

    loadConvLM(
        lmNetwork,
        lmCriterion,
        FLAGS_lmarchfile,
        FLAGS_lmweightfile,
        lmDict.indexSize(),
        adaptiveTail,
        FLAGS_lmadasoftmaxinputdim);

    if (lmCriterion) {
      auto as = std::static_pointer_cast<fl::AdaptiveSoftMaxLoss>(lmCriterion)
                    ->getActivation();
      std::dynamic_pointer_cast<fl::Sequential>(lmNetwork)->add(as);
    }
  }

  return lmNetwork;
}

std::shared_ptr<LMCritic> createLMCritic(
    const Dictionary& lmDict,
    const Dictionary& amDict) {
  auto lmNetwork = initLM(lmDict);
  std::vector<int> dictIndexMap;
  int numDictPadding;
  std::tie(dictIndexMap, numDictPadding) = genTokenDictIndexMap(lmDict, amDict);

  auto lmcrit = std::make_shared<LMCritic>(
      lmNetwork,
      dictIndexMap,
      numDictPadding,
      lmDict.getIndex(kEosToken),
      lmDict.getIndex(kUnkToken),
      FLAGS_gumbel,
      FLAGS_gumbeltemperature);

  return lmcrit;
}
} // namespace w2l
