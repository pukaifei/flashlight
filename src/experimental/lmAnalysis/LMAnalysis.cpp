/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <flashlight/flashlight.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lListFilesDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "module/module.h"
#include "runtime/Serial.h"
#include "experimental/localPriorMatch/src/runtime/Defines.h"
#include "experimental/localPriorMatch/src/runtime/Init.h"
#include "experimental/localPriorMatch/src/runtime/Utils.h"
#include "experimental/localPriorMatch/src/module/LMCritic.h"

using namespace w2l;

float compute_lmcrit_score(
    std::vector<std::vector<int>> paths, 
    Dictionary& dict, 
    std::shared_ptr<LMCritic> lmcrit) {
  auto target = batchTarget(paths, dict.getIndex(kEosToken));
  auto targetOnehot = fl::Variable(makeOnehot(target, dict.indexSize()), false);
  auto lmcrit_loss = lmcrit->forward({fl::log(targetOnehot)}).front();
  // af::print("LM Loss", lmcrit_loss.array());
  return (-1 * lmcrit_loss).scalar<float>();
}

template <typename T, typename S>
float compute_error_rate(
    const T& tgt, const S& hyp, fl::EditDistanceMeter& meter) {
  meter.reset();
  meter.add(hyp, tgt);
  return meter.value()[0];
}

std::vector<float> compute_error_rates(
    std::vector<int> tgtTkn, std::vector<int> hypTkn, Dictionary& dict,
    fl::EditDistanceMeter& meter) {
  remapLabels(tgtTkn, dict);
  remapLabels(hypTkn, dict);
  auto tgtLtr = tknIdx2Ltr(tgtTkn, dict);
  auto hypLtr = tknIdx2Ltr(hypTkn, dict);
  auto tgtWrd = split(FLAGS_wordseparator, stringify<std::string>(tgtLtr, ""), true);
  auto hypWrd = split(FLAGS_wordseparator, stringify<std::string>(hypLtr, ""), true);
  
  auto ter = compute_error_rate(tgtTkn, hypTkn, meter);
  auto ler = compute_error_rate(tgtLtr, hypLtr, meter);
  auto wer = compute_error_rate(tgtWrd, hypWrd, meter);
  return {ter, ler, wer};
}

std::vector<size_t> get_lengths(std::vector<int> tkn, Dictionary& dict) {
  remapLabels(tkn, dict);
  auto ltr = tknIdx2Ltr(tkn, dict);
  auto wrd = split(FLAGS_wordseparator, stringify<std::string>(ltr, ""), true);
  return {tkn.size(), ltr.size(), wrd.size()};
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec +
      " [model] [decodeResult] [analysisResult]");

  if (argc <= 2) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  // network and criterion will not be used. used for serializing
  std::string reloadpath = argv[1];
  std::string inpPath = argv[2];
  std::string outPath = argv[3];
  std::unordered_map<std::string, std::string> cfg;
  std::shared_ptr<fl::Module> base_network;
  std::shared_ptr<SequenceCriterion> base_criterion;

  std::ofstream outStream(outPath);
  if (!outStream.is_open() || !outStream.good()) {
    LOG(FATAL) << "Error opening analysis file: " << outPath;
  }
  std::ifstream inpStream(inpPath);
  if (!inpStream.is_open() || !inpStream.good()) {
    LOG(FATAL) << "Error opening decode result file: " << inpPath;
  }

  W2lSerializer::load(reloadpath, cfg, base_network, base_criterion);

	auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "Invalid config loaded from " << reloadpath;
  }
  LOG(INFO) << "Reading flags from config file " << reloadpath;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

	/* =============== Create Dictionary and Lexicon ================ */
  Dictionary dict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  // Setup-specific modifications
  if (FLAGS_eostoken) {
    dict.addEntry(kEosToken);
  }

  int numClasses = dict.indexSize();
  dict.setDefaultIndex(numClasses); 
  LOG(INFO) << "Number of classes (network) = " << dict.indexSize();

	// /* =============== Create LMCritic and Reload =================== */
	Dictionary lmDict = createFairseqTokenDict(FLAGS_lmdict);
  LOG(INFO) << "Number of classes (lm) = " << lmDict.indexSize();

  std::shared_ptr<LMCritic> lmcrit;
	lmcrit = createLMCritic(lmDict, dict);
  lmcrit->eval();
  LOG(INFO) << "[LMCritic] " << lmcrit->prettyString();
  LOG(INFO) << "[LMCritic Params] " << numTotalParams(lmcrit);
  
	/* =========== Create Meters and Helper Functions =============== */
  af::setMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  std::string metername = FLAGS_target == "ltr" ? "LER: " : "PER: ";

  fl::EditDistanceMeter cerMeter_single;

	/* ============== Enumerate Through DecodeResults  ==================== */
  std::string line;
  std::vector<int> tgtraw;
  std::vector<float> errs;
  int uid = 1;
  
  // TODO: debug batched LM
  // int batchSz = 4;
  // std::vector<std::vector<int>> tokensBatch;
  while (std::getline(inpStream, line)) {
    // LOG(INFO) << "[ " << uid << " ] LINE: " << line;

    auto fields = split(' ', line, true);
    if ((fields[0] != "Tgt" && fields[0] != "Hyp") || fields.size() != 4) {
      continue;
    } 

    std::vector<int> tokens;
    tokens.clear();
    for (auto s : split(',', fields[3], true)) {
      tokens.push_back(std::stoi(s));
    }
    auto tokens_lm = remapLabelsForLM(tokens, dict);
    
    if (fields[0] == "Tgt") {
      tgtraw = tokens;
      errs = {0, 0, 0};
      uid++;
    } else {
      errs  = compute_error_rates(tgtraw, tokens, dict, cerMeter_single);
    }

    auto lengths = get_lengths(tokens, dict);
    auto lm_score = compute_lmcrit_score({tokens_lm}, dict, lmcrit);
    
    // TODO: debug batched LM
    // tokensBatch.push_back(tokens_lm);
    // if (tokensBatch.size() == batchSz) {
    //   std::cout << "============ Print Tokens Batch\n";
    //   auto batch_lm_score = compute_lmcrit_score(tokensBatch, dict, lmcrit);
    //   tokensBatch.resize(0);
    //   std::cout << "===============================\n";
    // }

    if (fields[0] == "Tgt") {
      outStream << "===============" << std::endl;
    }
    outStream << fields[0] << " " << fields[1] << " " << fields[2]
              << " LM_Score=" << lm_score
              << " TER=" << errs[0]
              << " LER=" << errs[1]
              << " WER=" << errs[2]
              << " TL=" << lengths[0]
              << " LL=" << lengths[1]
              << " WL=" << lengths[2]
              << " Txt=" << stringify<std::string>(wrdIdx2Wrd(tokens_lm, dict))
              << std::endl;
  }
  return 0;
}
