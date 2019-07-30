/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <vector>

#include <flashlight/contrib/contrib.h>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Utils.h"
#include "libraries/lm/ConvLM.h"
#include "libraries/lm/KenLM.h"
#include "module/module.h"
#include "runtime/runtime.h"

namespace {
DEFINE_bool(
    usespelling,
    false,
    "Compute token level LM scores using the provided lexicon");
DEFINE_string(sentence, "", "Single input sentence to analysis LM score");
DEFINE_string(
    sentencesfile,
    "",
    "A list of input sentences to analysis LM score");
DEFINE_string(resultfile, "", "output of the analysis");
DEFINE_bool(verbose, false, "Print LM of each word in the sentence");
} // namespace

using namespace w2l;

std::vector<float> getLmScore(
    const std::vector<std::string>& sentence,
    const std::shared_ptr<LM>& lm,
    const Dictionary& dict) {
  int sentenceLength = sentence.size();
  std::vector<float> scores(sentenceLength + 1);

  auto inState = lm->start(0);
  for (int i = 0; i < sentenceLength; i++) {
    const auto& word = sentence[i];
    auto lmReturn = lm->score(inState, dict.getIndex(word));
    inState = lmReturn.first;
    scores[i] = lmReturn.second;
    lm->updateCache({inState});
  }
  auto lmReturn = lm->finish(inState);
  scores[sentenceLength] = lmReturn.second;
  return scores;
}

std::string processOneSentence(
    const std::string& sentence,
    const std::shared_ptr<LM>& lm,
    const Dictionary& dict,
    const LexiconMap& lexicon) {
  auto splitSentence = splitOnWhitespace(sentence, true);
  if (FLAGS_usespelling) {
    for (auto s : splitSentence) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
    splitSentence = wrd2Target(splitSentence, lexicon, dict, true, true);
    for (auto s : splitSentence) {
      std::cout << s << " ";
    }
    std::cout << std::endl;
  }
  auto scores = getLmScore(splitSentence, lm, dict);

  std::stringstream buffer;
  if (FLAGS_verbose) {
    buffer << "  - (</s>, 0)" << std::endl;
    for (int i = 0; i < splitSentence.size(); i++) {
      buffer << "  - (" << splitSentence[i] << ", " << std::setprecision(5)
             << scores[i] << ")" << std::endl;
    }
    buffer << "  - (</s>, " << std::setprecision(5) << scores.back() << ")"
           << std::endl;
  }

  float sum = std::accumulate(scores.begin(), scores.end(), 0.);
  std::string joinedSentence = join(" ", splitSentence);
  buffer << std::setprecision(5) << sum << '\t' << joinedSentence << "\n---"
         << std::endl;

  return buffer.str();
}

/**
 * Three use cases:
 * 1. word sentence + word LM + lexicon
 * 2. word sentence + token LM + lexicon + token list
 * 3. token sentence + token LM + token list
 *
 */
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: " + exec + " \n Compulsory: [lm_flags] [dictionary_flags] " +
      "\n Optional: [-sentencesfile] [-sentence] [-verbose] [-resultfile]");
  if (argc <= 1) {
    throw std::invalid_argument(gflags::ProgramUsage());
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  /* ===================== Build Dictionary ===================== */
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  }

  Dictionary usrDict;
  if (!FLAGS_lexicon.empty() && !FLAGS_usespelling) {
    // Case 1
    usrDict = createWordDict(lexicon);
  } else {
    // Case 2, 3
    auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
    Dictionary tokenDict(dictPath);
    for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
      tokenDict.addEntry(std::to_string(r));
    }
    usrDict = tokenDict;
  }
  std::cerr << "Number of words: " << usrDict.indexSize();

  /* ===================== Build LM ===================== */
  std::shared_ptr<LM> lm;
  if (FLAGS_lmtype == "kenlm") {
    lm = std::make_shared<KenLM>(FLAGS_lm, usrDict);
    if (!lm) {
      throw std::runtime_error(
          "[LM constructing] Failed to load LM: " + FLAGS_lm);
    }
  } else if (FLAGS_lmtype == "convlm") {
    af::setDevice(0);
    std::shared_ptr<fl::Module> convLmModel;
    W2lSerializer::load(FLAGS_lm, convLmModel);
    convLmModel->eval();

    auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
    lm = std::make_shared<ConvLM>(
        getConvLmScoreFunc,
        FLAGS_lm_vocab,
        usrDict,
        FLAGS_lm_memory,
        FLAGS_beamsize);
  } else {
    throw std::runtime_error(
        "[LM constructing] Invalid LM Type: " + FLAGS_lmtype);
  }

  /* ===================== Analysis ===================== */
  if (!FLAGS_sentence.empty()) {
    std::cout << processOneSentence(FLAGS_sentence, lm, usrDict, lexicon);
  }

  if (!FLAGS_sentencesfile.empty()) {
    if (!fileExists(FLAGS_sentencesfile)) {
      throw std::invalid_argument(
          "Dictionary file '" + FLAGS_sentencesfile + "' does not exist.");
    }
    std::ifstream inStream(FLAGS_sentencesfile);

    std::ofstream outStream;
    if (!FLAGS_resultfile.empty()) {
      outStream.open(FLAGS_resultfile);
      if (!outStream.is_open() || !outStream.good()) {
        throw std::runtime_error(
            "Error opening result file: " + FLAGS_resultfile);
      }
    }

    std::string line, result;
    while (std::getline(inStream, line)) {
      if (line.empty()) {
        continue;
      }
      result = processOneSentence(line, lm, usrDict, lexicon);

      std::cout << result;
      if (!FLAGS_resultfile.empty()) {
        outStream << result;
      }
    }
  }

  return 0;
}
