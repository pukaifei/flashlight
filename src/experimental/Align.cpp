/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "data/W2lDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "fb/W2lEverstoreDataset.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include "experimental/AlignUtils.h"
#include "experimental/ForceAlignment.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: \n " + exec + " align_file_path [flags]");
  if (argc <= 2) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::string alignFilePath = argv[1];

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
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
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  Dictionary tokenDict(FLAGS_tokens);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  if (FLAGS_eostoken) {
    tokenDict.addEntry(kEosToken);
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, tokenDict});

  std::mutex write_mutex;
  std::ofstream alignFile;
  alignFile.open(alignFilePath);
  if (!alignFile.is_open() || !alignFile.good()) {
    LOG(FATAL) << "Error opening log file";
  }

  auto writeLog = [&](const std::string& logStr) {
    std::lock_guard<std::mutex> lock(write_mutex);
    alignFile << logStr;
  };

  /* ===================== Create Dataset ===================== */
  int worldRank = 0;
  int worldSize = 1;
  std::shared_ptr<W2lDataset> ds;
  if (FLAGS_everstoredb) {
    W2lEverstoreDataset::init(); // Required for everstore client
    LexiconMap dummyLexicon;
    ds = std::make_shared<W2lEverstoreDataset>(
        FLAGS_test,
        dicts,
        dummyLexicon,
        FLAGS_batchsize,
        worldRank,
        worldSize,
        true /* fallback2Ltr */,
        false /* skipUnk */,
        FLAGS_datadir);
  } else {
    ds = std::make_shared<W2lNumberedFilesDataset>(
        FLAGS_test,
        dicts,
        FLAGS_batchsize,
        worldRank,
        worldSize,
        FLAGS_datadir);
  }

  LOG(INFO) << "[Dataset] Dataset loaded.";

  auto transition = afToVector<float>(criterion->params()[0].array());
  w2l::ForceAlignment fa(transition);

  int batches = 0;
  fl::TimeMeter alignMtr;
  fl::TimeMeter fwdMtr;
  fl::TimeMeter parseMtr;

  for (auto& sample : *ds) {
    fwdMtr.resume();
    auto rawEmission = network->forward({fl::input(sample[kInputIdx])}).front();
    fwdMtr.stop();
    alignMtr.resume();
    auto bestPaths = fa.align(rawEmission, fl::input(sample[kTargetIdx]));
    alignMtr.stop();

    parseMtr.resume();
#pragma omp parallel for num_threads(bestPaths.size())
    for (auto b = 0; b < bestPaths.size(); b++) {
      auto sampleIdsStr = readSampleIds(sample[kSampleIdx]);
      auto rawLtrTarget = afToVector<int>(sample[kTargetIdx]);
      for (auto& t : rawLtrTarget) {
        // ignore padded letter targets
        if (t == -1) {
          break;
        }
      }

      std::vector<std::string> alignedLetters;
      auto path = bestPaths[b];
      for (auto& p : path) {
        auto ltr = dicts[kTargetIdx].getEntry(p);
        alignedLetters.emplace_back(ltr);
      }
      auto alignedWords = getAlignedWords(alignedLetters, FLAGS_replabel);

      // write alignment output to align file in CTM format
      if (sampleIdsStr.size() > b) {
        auto ctmString = getCTMFormat(alignedWords);
        std::stringstream buffer;
        buffer << sampleIdsStr[b] << "\t" << ctmString << "\n";
        writeLog(buffer.str());
      }
    }
    parseMtr.stop();
    ++batches;
    if (batches % 500 == 0) {
      LOG(INFO) << "Done batches: " << batches
                << " , samples: " << batches * FLAGS_batchsize;
    }
  }

  LOG(INFO) << "Align time: " << alignMtr.value();
  LOG(INFO) << "Fwd time: " << fwdMtr.value();
  LOG(INFO) << "Parse time: " << parseMtr.value();
  alignFile.close();
  return 0;
}
