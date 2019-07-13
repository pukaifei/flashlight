/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "experimental/semisupervised/src/module/LMCritic.h"
#include "experimental/semisupervised/src/runtime/runtime.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + std::string() +
      " continue [directory] [flags]\n or " + std::string(argv[0]) +
      " fork [directory/model] [flags]");

  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  auto config = setFlags(argc, argv);

  int runIdx = std::stoi(config[kRunIdx]);
  std::string reloadPath = config[kReloadPath];
  int startEpoch = std::stoi(config[kStartEpoch]);
  int startIter = std::stoi(config[kStartIter]);
  std::string runPath = config[kRunPath];
  std::string runStatus = config[kRunStatus];

  /* ================ Set up distributed environment ================ */
  af::setMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    initDistributed(FLAGS_world_rank, FLAGS_world_size, FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(
        1.0 / fl::getWorldSize(), true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  LOG_MASTER(INFO) << "Experiment path: " << runPath;
  LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  /* ===================== Create Dictionary & Lexicon ===================== */
  Dictionary dict(FLAGS_tokens);
  // Setup-specific modifications
  if (FLAGS_eostoken) {
    dict.addEntry(kEosToken);
  }

  int numClasses = dict.indexSize();
  dict.setDefaultIndex(numClasses);
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);

  Dictionary lmDict = createFairseqTokenDict(FLAGS_lmdict);

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<Seq2SeqCriterion> criterion;
  std::shared_ptr<LMCritic> lmcrit;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;

  if (runStatus == kTrainMode) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();

    network = createW2lSeqModule(archfile, numFeatures, numClasses);
    criterion = std::make_shared<Seq2SeqCriterion>(
        buildSeq2Seq(numClasses, dict.getIndex(kEosToken)));
    lmcrit = createLMCritic(lmDict, dict);
  } else {
    std::unordered_map<std::string, std::string> cfg; // unused
    std::shared_ptr<SequenceCriterion> base_criterion;
    if (runStatus == kForkAMMode) {
      W2lSerializer::load(reloadPath, cfg, network, base_criterion, netoptim);
      lmcrit = createLMCritic(lmDict, dict);
    } else {
      W2lSerializer::load(
          reloadPath, cfg, network, base_criterion, netoptim, lmcrit);
    }
    criterion = std::dynamic_pointer_cast<Seq2SeqCriterion>(base_criterion);
  }

  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();
  LOG_MASTER(INFO) << "[Criterion Params: " << numTotalParams(criterion) << "]";
  LOG_MASTER(INFO) << "[LMCritic] " << lmcrit->prettyString();
  LOG_MASTER(INFO) << "[LMCritic Params: " << numTotalParams(lmcrit) << "]";

  if (runStatus != kContinueMode) {
    netoptim = initOptimizer(
        {network, criterion},
        FLAGS_netoptim,
        FLAGS_lr,
        FLAGS_momentum,
        FLAGS_weightdecay);
  }
  LOG_MASTER(INFO) << "[Optimizer] " << netoptim->prettyString();

  /* ===================== Create Dataset ===================== */
  auto pairedDs = createDataset(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  auto unpairedAudioDs = createDataset(
      FLAGS_trainaudio, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    pairedDs->shuffle(FLAGS_seed);
    unpairedAudioDs->shuffle(FLAGS_seed);
  }

  auto trainEvalIds =
      getTrainEvalIds(pairedDs->size(), FLAGS_pcttraineval, FLAGS_seed);

  auto validSets = split(',', trim(FLAGS_valid));
  std::unordered_map<std::string, std::shared_ptr<W2lDataset>> validds;
  for (const auto& s : validSets) {
    auto ts = splitOnAnyOf(":", s);
    auto setKey = ts.size() == 1 ? s : ts[0];
    auto setValue = ts.size() == 1 ? s : ts[1];

    validds[setKey] = createDataset(
        setValue, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  }

  /* ===================== Training Dataset Scheduler ===================== */
  DataScheduler trainDscheduler(
      {pairedDs, unpairedAudioDs},
      {kParallelData, kUnpairedAudio},
      {FLAGS_pairediter, FLAGS_audioiter},
      startEpoch + 1);

  int64_t nItersPerEpoch = FLAGS_pairediter + FLAGS_audioiter;

  /* ===================== Meters ===================== */
  SSLTrainMeters meters;
  for (const auto& s : validds) {
    meters.valid[s.first] = SSLDatasetMeters();
  }
  resetTimeStatMeters(meters);
  resetDatasetMeters(meters.train);

  /* ===================== Logging ===================== */
  bool logOnEpoch = FLAGS_reportiters == 0;
  LogHelper logHelper(runIdx, runPath, isMaster, logOnEpoch);
  logHelper.saveConfig(config);
  logHelper.writeHeader(meters);

  /* ===================== Hooks ===================== */
  if (reducer) {
    fl::distributeModuleGrads(network, reducer);
    fl::distributeModuleGrads(criterion, reducer);
  }

  fl::allReduceParameters(network);
  fl::allReduceParameters(criterion);

  auto train = [&meters,
                &trainEvalIds,
                &trainDscheduler,
                &validds,
                &startEpoch,
                &startIter,
                &nItersPerEpoch,
                &network,
                &criterion,
                &lmcrit,
                &netoptim,
                &dicts,
                &config,
                &logHelper,
                &logOnEpoch,
                &reducer](int nEpochs) {
    int64_t curEpoch = startEpoch;
    int64_t curIter = startIter;
    bool isPairedData;
    network->train();
    criterion->train();
    lmcrit->eval();

    while (curEpoch < nEpochs) {
      double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
      netoptim->setLr(lrScale * FLAGS_lr);

      double lmTempScale =
          std::pow(FLAGS_gamma, curEpoch / FLAGS_lmtempstepsize);

      ++curEpoch;
      af::sync();
      meters.timer[kSampleTimer].resume();
      meters.timer[kRuntime].resume();
      meters.timer[kTimer].resume();
      LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";

      // linearly warm up the amount of unpaired audio data used in training
      if (FLAGS_audiowarmupepochs > 0 && curEpoch > FLAGS_pretrainWindow &&
          (curEpoch - FLAGS_pretrainWindow) <= FLAGS_audiowarmupepochs) {
        int unpairedIter = (curEpoch - FLAGS_pretrainWindow) * FLAGS_audioiter /
            FLAGS_audiowarmupepochs;
        trainDscheduler.setSchedule({FLAGS_pairediter, unpairedIter});
        nItersPerEpoch = FLAGS_pairediter + unpairedIter;
      }

      int scheduleIter = 0;
      while (scheduleIter < nItersPerEpoch) {
        auto sample = trainDscheduler.get();
        isPairedData = af::allTrue<bool>(sample[kDataTypeIdx] == kParallelData);
        ++curIter;
        ++scheduleIter;
        af::sync();

        meters.timer[kTimer].incUnit();
        meters.timer[kSampleTimer].stopAndIncUnit();
        meters.stats.add(sample[kInputIdx], sample[kTargetIdx]);
        if (af::anyTrue<bool>(af::isNaN(sample[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(sample[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values";
        }

        // forward
        meters.timer[kFwdTimer].resume();
        auto output = network->forward({fl::input(sample[kInputIdx])}).front();
        af::sync();

        meters.timer[kCritFwdTimer].resume();
        if (isPairedData) {
          criterion->setSampling(
              FLAGS_samplingstrategy, FLAGS_pctteacherforcing);
        } else { // isUnpairedAudio
          criterion->setSampling(FLAGS_unpairedSampling, -1);
          criterion->setGumbelTemperature(
              lmTempScale * FLAGS_gumbeltemperature);
        }

        // For unpaired audio, we currently use the target length to determine
        // the number of decoding steps, but we won't use the loss for training.
        auto critFwd =
            criterion->forward({output, fl::noGrad(sample[kTargetIdx])});
        auto s2sLoss = critFwd[0];
        auto s2sLogProb = critFwd[1];
        af::sync();
        meters.timer[kCritFwdTimer].stopAndIncUnit();

        fl::Variable loss;
        if (isPairedData) {
          if (af::anyTrue<bool>(af::isNaN(s2sLoss.array()))) {
            LOG(FATAL) << "ASR loss has NaN values";
          }
          meters.train.losses[kASR].add(s2sLoss.array());
          loss = s2sLoss;
        } else {
          meters.timer[kLMCritFwdTimer].resume();
          auto lmcritLoss = lmcrit->forward({s2sLogProb}).front();
          af::sync();
          meters.timer[kLMCritFwdTimer].stopAndIncUnit();

          if (af::anyTrue<bool>(af::isNaN(lmcritLoss.array()))) {
            LOG(FATAL) << "LMCritic loss has NaN values";
          }
          meters.train.losses[kLM].add(lmcritLoss.array());
          loss = FLAGS_lmweight * lmcritLoss;
        }
        af::sync();
        meters.timer[kFwdTimer].stopAndIncUnit();
        meters.train.losses[kFullModel].add(loss.array());

        // compute training error rate from parallel data
        if (isPairedData) {
          auto globalBatchIdx = afToVector<int64_t>(sample[kGlobalBatchIdx]);
          if (trainEvalIds.find(globalBatchIdx[0]) != trainEvalIds.end()) {
            evalOutput(
                output.array(),
                sample[kTargetIdx],
                meters.train.edits[kTarget],
                dicts[kTargetIdx],
                criterion);
          }
        }

        // backward
        meters.timer[kBwdTimer].resume();
        netoptim->zeroGrad();
        lmcrit->zeroGrad();
        loss.backward();
        if (reducer) {
          reducer->finalize();
        }

        af::sync();
        meters.timer[kBwdTimer].stopAndIncUnit();
        meters.timer[kOptimTimer].resume();

        // scale down gradients by batchsize
        for (const auto& p : network->params()) {
          p.grad() = p.grad() / FLAGS_batchsize;
        }
        for (const auto& p : criterion->params()) {
          p.grad() = p.grad() / FLAGS_batchsize;
        }
        if (FLAGS_maxgradnorm > 0) {
          auto params = network->params();
          auto critparams = criterion->params();
          params.insert(params.end(), critparams.begin(), critparams.end());
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }
        netoptim->step();
        af::sync();
        meters.timer[kOptimTimer].stopAndIncUnit();
        meters.timer[kSampleTimer].resume();

        // checkpoint evaluation
        if ((!logOnEpoch && curIter % FLAGS_reportiters == 0) ||
            (logOnEpoch && scheduleIter == nItersPerEpoch)) {
          stopTimeMeters(meters);
          runEval(
              network, criterion, lmcrit, validds, meters, dicts[kTargetIdx]);

          config[kEpoch] = std::to_string(curEpoch);
          config[kIteration] = std::to_string(curIter);
          std::unordered_map<std::string, double> logFields(
              {{"lr", netoptim->getLr()},
               {"lmcrit-t", lmTempScale * FLAGS_gumbeltemperature}});
          logHelper.logAndSaveModel(
              meters, config, network, criterion, lmcrit, netoptim, logFields);

          resetDatasetMeters(meters.train);
          resetTimeStatMeters(meters);
          network->train();
          criterion->train();
          meters.timer[kSampleTimer].resume();
          meters.timer[kRuntime].resume();
          meters.timer[kTimer].resume();
        }
      }
      af::sync();
    }

    startEpoch = curEpoch;
    startIter = curIter;
  };

  /* ===================== Training starts ===================== */
  if (FLAGS_pretrainWindow - startEpoch > 0) {
    nItersPerEpoch = pairedDs->size();
    trainDscheduler.setSchedule({pairedDs->size(), 0});
    train(FLAGS_pretrainWindow);
    auto s2s = std::dynamic_pointer_cast<Seq2SeqCriterion>(criterion);
    s2s->clearWindow();
    nItersPerEpoch = FLAGS_pairediter + FLAGS_audioiter;
    trainDscheduler.setSchedule({FLAGS_pairediter, FLAGS_audioiter});
    LOG_MASTER(INFO) << "Finished pretraining";
  }

  train(FLAGS_iter);

  LOG_MASTER(INFO) << "Finished training";
  return 0;
}
