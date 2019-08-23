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
#include "experimental/localPriorMatchOss/src/module/LMCritic.h"
#include "experimental/localPriorMatchOss/src/runtime/runtime.h"
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
  auto config = setFlags(argc, argv, false);

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
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
  }
  Dictionary dict(dictPath);
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
      FLAGS_trainaudio,
      dicts,
      lexicon,
      FLAGS_unpairedBatchsize,
      worldRank,
      worldSize);

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
      LOG_MASTER(INFO) << "  Learning rate = " << lrScale * FLAGS_lr;

      // linearly warm up the amount of unpaired audio data used in training
      if (FLAGS_audiowarmupepochs > 0 && curEpoch > FLAGS_pretrainWindow &&
          (curEpoch - FLAGS_pretrainWindow) <= FLAGS_audiowarmupepochs) {
        int unpairedIter = (curEpoch - FLAGS_pretrainWindow) * FLAGS_audioiter /
            FLAGS_audiowarmupepochs;
        trainDscheduler.setSchedule({FLAGS_pairediter, unpairedIter});
        nItersPerEpoch = FLAGS_pairediter + unpairedIter;
      }

      std::vector<std::vector<int>> paths;
      std::vector<int> hypoNums;
      int scheduleIter = 0;
      while (scheduleIter < nItersPerEpoch) {
        auto sample = trainDscheduler.get();
        isPairedData = af::allTrue<bool>(sample[kDataTypeIdx] == kParallelData);
        ++curIter;
        ++scheduleIter;
        af::sync();
        paths.clear();
        hypoNums.clear();
        int bs  = isPairedData ? FLAGS_batchsize : FLAGS_unpairedBatchsize;

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
        fl::Variable lment;
        if (isPairedData) {
          meters.timer[kCritFwdTimer].resume();
          loss = criterion->forward(
              {output, fl::noGrad(sample[kTargetIdx])}).front();
          
          if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
            LOG(FATAL) << "ASR loss has NaN values";
          }
          meters.train.losses[kASR].add(loss.array());
          meters.timer[kCritFwdTimer].stopAndIncUnit();
        } else {
          fl::Variable lmLogprob;
          meters.timer[kBeamTimer].resume();
          std::tie(paths, hypoNums) = batchBeamSearch(output, criterion);
          meters.timer[kBeamTimer].stopAndIncUnit();
          
          auto refLen = afToVector<int>(getTargetLength(
              sample[kTargetIdx], dicts[kTargetIdx].getIndex(kEosToken)));
          std::tie(paths, hypoNums) = filterBeamByLength(paths, hypoNums, refLen);
          auto hypoNumsArr = af::array(af::dim4(hypoNums.size()), hypoNums.data());
          af::array remIdx = af::sort(af::where(hypoNumsArr));
          int remBs = remIdx.dims()[0];

          if (remBs == 0) {
            LOG(INFO) << "WARNING : using a made-up loss because remBs=0";
            // create a made-up loss with 0 value that is a function of
            // parameters to train, so the grad will be all 0.
            loss = criterion->forward(
                {output, fl::noGrad(sample[kTargetIdx])}).front();
            loss = 0.0 * loss;
          } else {
            output = output(af::span, af::span, remIdx);
            hypoNums = afToVector<int>(hypoNumsArr(remIdx));

            meters.timer[kLMCritFwdTimer].resume();
            lmLogprob = computeLmLogprob(paths, lmcrit, dicts[kTargetIdx]);
            meters.timer[kLMCritFwdTimer].stopAndIncUnit();
            
            meters.timer[kBeamFwdTimer].resume();
            auto s2sLogprob = computeS2SLogprob(
                paths, hypoNums, output, criterion, dicts[kTargetIdx]);

            loss = computePriorMatchingLoss(lmLogprob, s2sLogprob, hypoNums);
            lment = entropy(lmLogprob, hypoNums);
            meters.timer[kBeamFwdTimer].stopAndIncUnit();

            for (auto& path : paths) {
              meters.train.losses[kLen].add(static_cast<double>(path.size()));
            }
            meters.train.losses[kNumHypos].add(static_cast<double>(paths.size()));
            meters.train.losses[kLMEnt].add(lment.array());
            meters.train.losses[kLMScore].add(lmLogprob.array());

            if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
              LOG(FATAL) << "LMCritic loss has NaN values";
            }
            meters.train.losses[kLM].add(loss.array());
            loss = FLAGS_lmweight * loss;
          }
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
                meters.train.edits,
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

        // scale down gradients by batchsize note that the original batchsize
        // bs is used instead of remBs, since different workers may have
        // different remBs. for the sake of simplicity we just use bs.
        for (const auto& p : network->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / bs;
        }
        for (const auto& p : criterion->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / bs;
        }
        // LOG_MASTER(INFO) << "clip grad";
        if (FLAGS_maxgradnorm > 0) {
          auto params = network->params();
          auto critparams = criterion->params();
          params.insert(params.end(), critparams.begin(), critparams.end());
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }
        // LOG_MASTER(INFO) << "step";
        netoptim->step();
        af::sync();
        meters.timer[kOptimTimer].stopAndIncUnit();
        meters.timer[kSampleTimer].resume();

        auto lengths = getLengths<int, int>(paths);
        LOG_MASTER(INFO) << "[ Epoch " << curEpoch << " ]" 
                         << " Iter=" << scheduleIter
                         << " isPairedData=" << isPairedData
                         << " AvgLoss=" << fl::mean(loss, {0}).scalar<float>()
                         << " MinLen=" << *std::min_element(lengths.begin(), lengths.end())
                         << " MaxLen=" << *std::max_element(lengths.begin(), lengths.end());


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
