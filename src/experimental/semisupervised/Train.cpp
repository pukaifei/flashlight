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
#include "common/Dictionary.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "experimental/semisupervised/runtime/runtime.h"
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
  Dictionary dict = createTokenDict();
  int numClasses = dict.indexSize();
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;

  if (runStatus == kTrainMode) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();

    network = createW2lSeqModule(archfile, numFeatures, numClasses);
    criterion = std::make_shared<Seq2SeqCriterion>(
        buildSeq2Seq(numClasses, dict.getIndex(kEosToken)));
  } else {
    std::unordered_map<std::string, std::string> cfg; // unused
    W2lSerializer::load(reloadPath, cfg, network, criterion, netoptim);
  }

  // TODO: load an LM module. We don't need to save LM-crit as it won't be used
  // during decoding, so always loading from file should be good enough for now.
  std::shared_ptr<fl::Module> lmcrit;

  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();
  LOG_MASTER(INFO) << "[Criterion Params: " << numTotalParams(criterion) << "]";

  if (runStatus == kTrainMode || runStatus == kForkMode) {
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
      randomSubset(FLAGS_seed, pairedDs->size(), FLAGS_pcttraineval);

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
      startEpoch);

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

    while (curEpoch < nEpochs) {
      double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
      netoptim->setLr(lrScale * FLAGS_lr);

      // TODO: support updating the iterations based on curEpoch
      // e.g. warming up #iterations for the unpaired audio set

      ++curEpoch;

      af::sync();
      meters.timer[kSampleTimer].resume();
      meters.timer[kRuntime].resume();
      meters.timer[kTimer].resume();
      LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
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

        fl::Variable loss;
        // ASR loss for parallel data
        if (isPairedData) {
          meters.timer[kCritFwdTimer].resume();
          loss = criterion->forward({output, fl::noGrad(sample[kTargetIdx])})
                     .front();
          af::sync();
          meters.timer[kCritFwdTimer].stopAndIncUnit();

          if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
            LOG(FATAL) << "ASR loss has NaN values";
          }
          meters.train.losses[kASR].add(loss.array());
        }

        // TODO: incorporate LM-crit loss, add timer,
        // add to both meters.train.losses[kLM] and
        // meters.train.losses[kFullModel] for logging
        meters.timer[kFwdTimer].stopAndIncUnit();

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
          runEval(network, criterion, validds, meters, dicts[kTargetIdx]);

          config[kEpoch] = std::to_string(curEpoch);
          config[kIteration] = std::to_string(curIter);
          logHelper.logAndSaveModel(
              meters, config, network, criterion, netoptim);

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
    trainDscheduler.setSchedule({pairedDs->size(), 0});
    train(FLAGS_pretrainWindow);
    auto s2s = std::dynamic_pointer_cast<Seq2SeqCriterion>(criterion);
    s2s->clearWindow();
    trainDscheduler.setSchedule({FLAGS_pairediter, FLAGS_audioiter});
    LOG_MASTER(INFO) << "Finished pretraining";
  }

  train(FLAGS_iter);

  LOG_MASTER(INFO) << "Finished training";
  return 0;
}
