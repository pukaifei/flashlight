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
//#include "runtime/runtime.h"
#include "criterion/TransformerCriterion.h"
#include "experimental/lead2Gold/src/runtime/runtime.h"
#include "runtime/SpeechStatMeter.h"

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
  std::shared_ptr<fl::Module> network_load;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;

  W2lSerializer::load(FLAGS_am, cfg, network_load, criterion);

  // std::shared_ptr<fl::Sequential> network = network_load;
  std::shared_ptr<fl::Sequential> network =
      std::dynamic_pointer_cast<fl::Sequential>(network_load);

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
  auto runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);

  auto filename_conf = getRunFile("config", 1, runPath);
  std::ofstream output_file_conf(filename_conf);
  output_file_conf.close();

  auto filename = getRunFile("perf", 1, runPath);
  std::ofstream output_file(filename);

  LOG(INFO) << "Logging in " << runPath;

  /* ===================== Create Dictionary ===================== */

  // auto tokenDict = createTokenDict();
  Dictionary tokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  FLAGS_replabel = 0;
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  if (FLAGS_criterion == kCtcCriterion ||
      FLAGS_criterion == kCtcBeamNoiseCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  Dictionary noiselmDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  noiselmDict.addEntry(kEosToken);

  DictionaryMap dicts = {{kCleanNoiselmKeyIdx, noiselmDict},
                         {kNoisyNoiselmKeyIdx, noiselmDict},
                         {kTargetIdx, tokenDict},
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

  FLAGS_criterion = kCtcCriterion;
  auto ds = createDatasetNoise(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  int nSamples = ds->size();
  ds->shuffle(FLAGS_seed);

  auto ds_valid =
      createDatasetNoise(FLAGS_valid, dicts, lexicon, 1, worldRank, worldSize);

  LOG(INFO) << "[Dataset] Dataset loaded.";

  // out : return Dim size of the output of the Nth layer.
  auto getdimX = [&](int N) {
    int out;
    auto sample = ds_valid->get(0);
    auto input = fl::input(sample[kInputIdx]);
    int num_layer = 1;
    // std::cout << "Nb of modules: " << network->modules().size() << std::endl;
    for (auto& module : network->modules()) {
      input = module->forward({input}).front();
      if (num_layer == N) {
        // std::cout << "DIMX = " << input.dims(0) << std::endl;
        return input.dims(0);
      }
      num_layer++;
    }
    return input.dims(0);
  };

  std::shared_ptr<EncDecCriterionMulti> encdecMulti =
      std::make_shared<EncDecCriterionMulti>(
          noiselmDict.indexSize(),
          getdimX(FLAGS_XFromLayerN),
          FLAGS_encoderdim,
          noiselmDict.getIndex(kEosToken),
          FLAGS_maxdecoderoutputlen,
          FLAGS_encoderrnnlayer,
          FLAGS_decoderrnnlayer,
          FLAGS_labelsmooth,
          FLAGS_pctteacherforcing,
          FLAGS_decoderdropout,
          FLAGS_decoderdropout,
          FLAGS_useSinPosEmb,
          FLAGS_usePosEmbEveryLayer);

  LOG_MASTER(INFO) << "[Enc/Dec] " << encdecMulti->prettyString();
  LOG_MASTER(INFO) << "[Enc/Dec Params: " << numTotalParams(encdecMulti) << "]";

  auto noiselmoptim = initOptimizer(
      {encdecMulti},
      FLAGS_netoptim,
      FLAGS_lr,
      FLAGS_momentum,
      FLAGS_weightdecay);

  if (reducer) {
    fl::distributeModuleGrads(encdecMulti, reducer);
  }
  fl::allReduceParameters(encdecMulti);

  // network = std::dynamic_pointer_cast<fl::Sequential>(network);
  network->eval();

  // evaluate first
  auto completeDS =
      [&](std::shared_ptr<NoiseW2lListFilesDataset> dsToComplete) {
        for (int64_t idx = 0; idx < dsToComplete->size(); idx++) {
          auto sample = dsToComplete->get(idx);
          auto output = network->forward(fl::input(sample[kInputIdx]));
          auto newTranscriptionsNoisy =
              getUpdateTrancriptsWords(output, criterion, dicts);
          dsToComplete->copyToGroundTruthTranscript(idx);
          dsToComplete->updateTargets(idx, newTranscriptionsNoisy);
        }
      };

  // transcripts T*B
  auto plotTranscript = [&](af::array& transcripts, Dictionary& dict) {
    for (int64_t b = 0; b < transcripts.dims(1); b++) {
      auto transcript_af = transcripts(af::span, b);
      auto transcript_raw = w2l::afToVector<int>(transcript_af);
      auto transcript_sz =
          w2l::getTargetSize(transcript_raw.data(), transcript_raw.size());
      transcript_raw.resize(transcript_sz);
      for (int j = 0; j < transcript_sz; j++) {
        std::cout << dict.getEntry(transcript_raw[j]);
      }
      std::cout << std::endl;
    }
  };

  // out[0] : output of the Nth layer. N start to 1.
  // out[1] : output of the entire network if returnlast is true
  auto forwardNLayer = [&](fl::Variable input, int N, bool returnlast) {
    std::vector<fl::Variable> out;
    int num_layer = 1;
    for (auto& module : network->modules()) {
      input = module->forward({input}).front();
      if (num_layer == N) {
        out.push_back(input);
        if (!returnlast) {
          break;
        }
      }
      num_layer++;
    }
    if (returnlast) {
      out.push_back(input);
    }
    return out;
  };

  auto mtr_TER_once = fl::EditDistanceMeter();

  // weighted Token Error Rate
  auto wTER = [&](af::array& clean_transcript,
                  std::vector<EncDecCriterionMulti::CandidateHypo>& Hypos) {
    float norm_ = 0;
    auto mtr_TER = fl::EditDistanceMeter();
    for (auto& Hypo : Hypos) {
      norm_ += std::exp(Hypo.score);
    }
    float wTER = 0;
    for (auto& Hypo : Hypos) {
      mtr_TER_once.reset();
      auto score = std::exp(Hypo.score) / norm_;
      auto path_af = af::array(Hypo.path.size(), Hypo.path.data());
      mtr_TER_once.add(clean_transcript, path_af);
      wTER += score * mtr_TER_once.value()[0];
    }
    return wTER;
  };

  std::cout << "GENERATING TRAIN NOISY TRANSCRIPTIONS..." << std::endl;
  completeDS(ds);
  std::cout << "DONE" << std::endl;
  std::cout << "GENERATING VALID NOISY TRANSCRIPTIONS..." << std::endl;
  completeDS(ds_valid);
  std::cout << "DONE" << std::endl;

  auto mtr_loss_train = fl::AverageValueMeter();
  auto mtr_loss_valid = fl::AverageValueMeter();

  auto timer_train_loop = fl::TimeMeter();
  auto timer_valid_loop = fl::TimeMeter();

  auto mtr_TER_true_noise_valid = fl::EditDistanceMeter();
  auto mtr_TER_true_recover_valid = fl::EditDistanceMeter();
  auto mtr_TER_true_recoverBeam_valid = fl::EditDistanceMeter();

  auto mtr_TER_true_noise_AVG_valid = fl::AverageValueMeter();
  auto mtr_TER_true_recover_AVG_valid = fl::AverageValueMeter();
  auto mtr_wTER_true_recoverBeam_AVG_valid = fl::AverageValueMeter();

  // write header
  auto header =
      "#\tepoch\ttrain_loss\ttrain_time\tvalid_loss\tvalid_time\tTER_base\tLER_recov\tLER_recovBeam\tavg_TER_base\tavg_LER_recov\twLER_recovBeam\tlr";
  w2l::appendToLog(output_file, header);
  int curEpoch = 0;
  int curIter = 0;
  std::string metrics;
  double initlr = noiselmoptim->getLr();

  while (curEpoch < FLAGS_iter) {
    std::string metrics = "";
    ++curEpoch;

    if (curEpoch >= FLAGS_lr_decay &&
        (curEpoch - FLAGS_lr_decay) % FLAGS_lr_decay_step == 0) {
      initlr /= 2;
    }

    encdecMulti->train();
    mtr_loss_train.reset();
    mtr_loss_valid.reset();

    timer_train_loop.reset();
    timer_valid_loop.reset();

    mtr_TER_true_noise_valid.reset();
    mtr_TER_true_recover_valid.reset();
    mtr_TER_true_recoverBeam_valid.reset();

    mtr_TER_true_noise_AVG_valid.reset();
    mtr_TER_true_recover_AVG_valid.reset();
    mtr_wTER_true_recoverBeam_AVG_valid.reset();

    for (auto& sample : *ds) {
      timer_train_loop.resume();
      ++curIter;
      auto noisyTarget = sample[kNoisyNoiselmKeyIdx];
      auto cleanTarget = sample[kCleanNoiselmKeyIdx];

      // Sometimes simply try to copy/past
      auto randNB = std::rand() % 101; // randNB in the range 0 to 100
      if (FLAGS_UseCopy > randNB && curIter < FLAGS_warmup) {
        cleanTarget = sample[kNoisyNoiselmKeyIdx];
      }

      noiselmoptim->setLr(
          initlr * std::min(curIter / double(FLAGS_warmup), 1.0));
      fl::Variable AcousticFeat =
          forwardNLayer(fl::input(sample[kInputIdx]), FLAGS_XFromLayerN, false)
              .front();

      auto res = encdecMulti->forward(
          {fl::input(noisyTarget), AcousticFeat, fl::noGrad(cleanTarget)});

      auto& loss = res[0];
      auto& out = res[1];

      mtr_loss_train.add(loss.array());

      loss.backward();
      if (reducer) {
        reducer->finalize();
      }
      af::sync();

      for (const auto& p : encdecMulti->params()) {
        if (!p.isGradAvailable()) {
          continue;
        }
        p.grad() = p.grad() / sample[kInputIdx].dims(3);
      }

      if (FLAGS_maxgradnorm > 0) {
        fl::clipGradNorm(encdecMulti->params(), FLAGS_maxgradnorm);
      }

      noiselmoptim->step();
      af::sync();

      timer_train_loop.stopAndIncUnit();
    }
    std::cout << "AVERAGE LOSS EPOCH " << curEpoch << ": "
              << mtr_loss_train.value()[0] << std::endl;
    std::cout << "DONE IN  " << timer_train_loop.value() << " ms/sample "
              << std::endl;
    metrics += std::to_string(curEpoch) + "\t" +
        std::to_string(mtr_loss_train.value()[0]);
    metrics += "\t" + std::to_string(timer_train_loop.value());
    // Evaluate model
    encdecMulti->eval();

    std::cout << "Evaluate on VALID" << std::endl;
    if (curEpoch % FLAGS_evaluateValidEveryNEpoch == 0) {
      for (auto& sample : *ds_valid) {
        timer_valid_loop.resume();
        auto& noisyTarget = sample[kNoisyNoiselmKeyIdx];
        auto cleanTarget = sample[kCleanNoiselmKeyIdx];
        fl::Variable AcousticFeat =
            forwardNLayer(
                fl::input(sample[kInputIdx]), FLAGS_XFromLayerN, false)
                .front();

        auto encodedx = encdecMulti->encodeT(fl::input(noisyTarget));
        af::array recoverTarget =
            encdecMulti->viterbiPathBase({encodedx, AcousticFeat}, true);

        auto loss_valid = encdecMulti
                              ->forward({fl::input(noisyTarget),
                                         AcousticFeat,
                                         fl::noGrad(cleanTarget)})
                              .front();
        mtr_loss_valid.add(loss_valid.array());

        // BEAM TESTING
        // auto beamRes = encdecMulti->beamSearchRes({encodedx, AcousticFeat},
        // FLAGS_beamsize, FLAGS_eosscore); auto beam_path = beamRes[0].path;
        // auto beam_path_af = af::array(beam_path.size(), beam_path.data());
        // auto wTER_res = wTER(cleanTarget, beamRes);

        std::cout << "clean        : ";
        plotTranscript(cleanTarget, dicts[kCleanNoiselmKeyIdx]);
        std::cout << "noisy        : ";
        plotTranscript(noisyTarget, dicts[kNoisyNoiselmKeyIdx]);
        std::cout << "recover      : ";
        plotTranscript(recoverTarget, dicts[kNoisyNoiselmKeyIdx]);
        // std::cout << "recover Beam : ";
        // plotTranscript(beam_path_af, dicts[kNoisyNoiselmKeyIdx]);

        mtr_TER_true_noise_valid.add(cleanTarget, noisyTarget);
        mtr_TER_true_recover_valid.add(cleanTarget, recoverTarget);
        // mtr_TER_true_recoverBeam_valid.add(cleanTarget, beam_path_af);

        mtr_TER_once.reset();
        mtr_TER_once.add(cleanTarget, noisyTarget);
        mtr_TER_true_noise_AVG_valid.add(mtr_TER_once.value()[0]);
        std::cout << "TER true<->noisy       : " << mtr_TER_once.value()[0]
                  << std::endl;

        mtr_TER_once.reset();
        mtr_TER_once.add(cleanTarget, recoverTarget);
        mtr_TER_true_recover_AVG_valid.add(mtr_TER_once.value()[0]);
        std::cout << "TER true<->recover     : " << mtr_TER_once.value()[0]
                  << std::endl;

        // mtr_TER_once.reset();
        // mtr_TER_once.add(cleanTarget, beam_path_af);
        // std::cout << "TER true<->recoverBEAM : " << mtr_TER_once.value()[0]
        // << std::endl;

        // mtr_wTER_true_recoverBeam_AVG_valid.add(wTER_res);
        // std::cout << "wTER true<->BEAM       : " << wTER_res << std::endl <<
        // std::endl;
        timer_valid_loop.stopAndIncUnit();
      }
    }

    std::cout << "     VALID  clean->noisy       : "
              << mtr_TER_true_noise_valid.value()[0] << std::endl;
    std::cout << "     VALID  clean->recover     : "
              << mtr_TER_true_recover_valid.value()[0] << std::endl;
    // std::cout << "     VALID  clean->recoverBEAM : " <<
    // mtr_TER_true_recoverBeam_valid.value()[0] << std::endl;

    std::cout << "AVG  VALID  clean->noisy       : "
              << mtr_TER_true_noise_AVG_valid.value()[0] << std::endl;
    std::cout << "AVG  VALID  clean->recover     : "
              << mtr_TER_true_recover_AVG_valid.value()[0] << std::endl;
    // std::cout << "wTER VALID  clean->recoverBEAM : " <<
    // mtr_wTER_true_recoverBeam_AVG_valid.value()[0] << std::endl;

    std::cout << "DONE IN  " << timer_valid_loop.value() << " ms/sample "
              << std::endl;
    std::cout << std::endl;

    metrics += "\t" + std::to_string(mtr_loss_valid.value()[0]);
    metrics += "\t" + std::to_string(timer_valid_loop.value());

    metrics += "\t" + std::to_string(mtr_TER_true_noise_valid.value()[0]);
    metrics += "\t" + std::to_string(mtr_TER_true_recover_valid.value()[0]);
    metrics += "\t" + std::to_string(mtr_TER_true_recoverBeam_valid.value()[0]);

    metrics += "\t" + std::to_string(mtr_TER_true_noise_AVG_valid.value()[0]);
    metrics += "\t" + std::to_string(mtr_TER_true_recover_AVG_valid.value()[0]);
    metrics +=
        "\t" + std::to_string(mtr_wTER_true_recoverBeam_AVG_valid.value()[0]);

    metrics += "\t" + std::to_string(noiselmoptim->getLr());
    w2l::appendToLog(output_file, metrics);

    ds->shuffle(curEpoch);
    ds_valid->shuffle(curEpoch);
  }
  output_file.close();
  return 0;
}
