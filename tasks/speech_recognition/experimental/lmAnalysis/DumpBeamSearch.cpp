/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/flashlight.h>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lListFilesDataset.h"
#include "experimental/localPriorMatch/src/module/LMCritic.h"
#include "experimental/localPriorMatch/src/runtime/Defines.h"
#include "experimental/localPriorMatch/src/runtime/Init.h"
#include "experimental/localPriorMatch/src/runtime/Utils.h"
#include "module/module.h"
#include "runtime/Serial.h"

using namespace w2l;

namespace w2l {
// outputs
DEFINE_bool(
    viewtranscripts,
    false,
    "Log the Reference and Hypothesis transcripts.");
DEFINE_string(attndir, "", "Directory for attention output.");

// decoding
DEFINE_int64(beamsz, 1, "Size of beam for beam search.");
} // namespace w2l

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec +
      " [model] [dataset] [output], optional: --attndir=[directory]");

  if (argc <= 3) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::string reloadpath = argv[1];
  std::string dataset = argv[2];
  std::string outpath = argv[3];
  std::unordered_map<std::string, std::string> cfg;
  std::shared_ptr<fl::Module> base_network;
  std::shared_ptr<SequenceCriterion> base_criterion;

  std::ofstream outStream;
  outStream.open(outpath);
  if (!outStream.is_open() || !outStream.good()) {
    LOG(FATAL) << "Error opening decode result file: " << outpath;
  }

  auto writeResult = [&](const std::string& type,
                         const std::vector<int>& path,
                         float score,
                         int rank) {
    outStream << type << " Rank=" << rank << " ASR_Score=" << score << " ";
    for (int i = 0; i < path.size(); i++) {
      if (i != 0) {
        outStream << ",";
      }
      outStream << path[i];
    }
    outStream << std::endl;
  };

  W2lSerializer::load(reloadpath, cfg, base_network, base_criterion);
  auto network = std::dynamic_pointer_cast<fl::Sequential>(base_network);
  auto criterion = std::dynamic_pointer_cast<Seq2SeqCriterion>(base_criterion);

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
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    dict.addEntry(std::to_string(r));
  }
  if (FLAGS_eostoken) {
    dict.addEntry(kEosToken);
  }

  LOG(INFO) << "Number of classes (network) = " << dict.indexSize();

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  std::shared_ptr<W2lDataset> testset;
  FLAGS_sampletarget = 0.0; // make sure to not sample wordpiece combinations
  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);

  testset = std::make_shared<W2lListFilesDataset>(
      dataset, dicts, lexicon, 1, 0, 1, true, true);

  /* ================= Print network information ================== */
  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Network Params] " << numTotalParams(network);
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Criterion Params] " << numTotalParams(criterion);

  /* =========== Create Meters and Helper Functions =============== */
  fl::afSetMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  std::string metername = FLAGS_target == "ltr" ? "LER: " : "PER: ";

  fl::EditDistanceMeter cerMeter_single;
  fl::EditDistanceMeter cerBeamMeter;
  fl::EditDistanceMeter werBeamMeter;
  fl::AverageValueMeter lossMeter;

  fl::TimeMeter beamSearchTimer(true);
  beamSearchTimer.reset();

  network->eval();
  criterion->eval();

  /* ============== Enumerate Through Dataset  ==================== */
  int uid = 1;
  for (auto& sample : *testset) {
    auto output = network->forward(fl::input(sample[kInputIdx]));
    auto target = sample[kTargetIdx];

    auto loss = criterion->forward({output, fl::noGrad(target)}).front();
    auto lossvec = afToVector<float>(loss.array());
    for (int b = 0; b < output.dims(2); ++b) {
      auto tgt = target(af::span, b);
      auto tgtraw = afToVector<int>(tgt);

      /* ===== Beam Search ===== */
      beamSearchTimer.resume();
      std::vector<Seq2SeqCriterion::CandidateHypo> beam;
      beam.emplace_back(Seq2SeqCriterion::CandidateHypo{});
      auto hypos = criterion->beamSearch(
          output.array()(af::span, af::span, b),
          beam,
          FLAGS_beamsz,
          FLAGS_maxdecoderoutputlen);
      auto beampath = hypos[0].path;
      beamSearchTimer.stopAndIncUnit();

      remapLabels(tgtraw, dict);
      auto score = -1 * loss(b).scalar<float>();
      writeResult("Tgt", tgtraw, score, -1);

      for (int r = 0; r < hypos.size(); ++r) {
        remapLabels(hypos[r].path, dict);
        writeResult("Hyp", hypos[r].path, hypos[r].score, r + 1);
      }

      remapLabels(beampath, dict);
      remapLabels(tgtraw, dict);

      auto beampathLtr = tknIdx2Ltr(beampath, dict);
      auto tgtrawLtr = tknIdx2Ltr(tgtraw, dict);

      cerBeamMeter.add(beampathLtr, tgtrawLtr);
      if (FLAGS_target == "ltr") {
        auto beamwords = split(
            FLAGS_wordseparator, stringify<std::string>(beampathLtr, ""), true);
        auto tgtwords = split(
            FLAGS_wordseparator, stringify<std::string>(tgtrawLtr, ""), true);
        werBeamMeter.add(beamwords, tgtwords);
      }
      lossMeter.add(lossvec[b]);

      if (FLAGS_viewtranscripts) {
        cerMeter_single.reset();
        cerMeter_single.add(beampathLtr, tgtrawLtr);

        std::cout << "UID: " << uid << ", " << metername
                  << cerMeter_single.value()[0]
                  << ", DEL: " << cerMeter_single.value()[2]
                  << ", INS: " << cerMeter_single.value()[3]
                  << ", SUB: " << cerMeter_single.value()[4] << std::endl;
        std::cout << "REF     ";
        std::cout << stringify<std::string>(tgtrawLtr, "") << std::endl;

        std::cout << "BM HYP  ";
        std::cout << stringify<std::string>(beampathLtr, "") << std::endl;
      }

      ++uid;
    }
  }

  auto beamcer = cerBeamMeter.value()[0];
  auto beamdel = cerBeamMeter.value()[2];
  auto beamins = cerBeamMeter.value()[3];
  auto beamsub = cerBeamMeter.value()[4];
  auto avgloss = lossMeter.value()[0];
  LOG(INFO) << "Beam Search " << metername << beamcer << ", DEL: " << beamdel
            << ", INS: " << beamins << ", SUB: " << beamsub;
  LOG(INFO) << "Teacher Forced " << metername << ", Loss: " << avgloss;

  if (FLAGS_target == "ltr") {
    auto beamwer = werBeamMeter.value()[0];
    LOG(INFO) << "Beam Search WER " << beamwer;
  }

  LOG(INFO) << "Avg Beam Search Time = " << beamSearchTimer.value();

  return 0;
}
