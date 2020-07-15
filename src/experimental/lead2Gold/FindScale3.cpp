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

#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "experimental/lead2Gold/src/common/Defines.h"
#include "experimental/lead2Gold/src/common/Utils.h"
#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "experimental/lead2Gold/src/data/Featurize.h"
#include "experimental/lead2Gold/src/data/Utils.h"
#include "experimental/lead2Gold/src/runtime/runtime.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"

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
  bool wasASGBeamNoise = config["wasASGBeamNoise"] == "1";
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

  LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  LOG_MASTER(INFO) << "Experiment path: " << runPath;
  LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
  }
  Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  // Setup-specific modifications
  if (FLAGS_eostoken) {
    tokenDict.addEntry(kEosToken);
  }

  int numClasses = tokenDict.indexSize();
  tokenDict.setDefaultIndex(numClasses);
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, tokenDict});

  Dictionary wordDict;
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
    dicts.insert({kWordIdx, wordDict});
  }

  /* =========== ASG BEAM NOISE SPECIFIC ============ */
  // to add LM look at wave2word code
  std::shared_ptr<NoiseTrie> noiselex = nullptr;
  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm;
  bool isNoiseModelTrained = false;
  Dictionary noise_keys(dictPath);
  if (FLAGS_criterion == kAsgBeamNoiseCriterion) {
    if (FLAGS_uselexicon && !FLAGS_lexicon.empty()) {
      noiselex = std::shared_ptr<NoiseTrie>(new NoiseTrie(
          tokenDict.indexSize() - FLAGS_replabel,
          tokenDict.getIndex("|"),
          nullptr));
      auto words = noiselex->load(FLAGS_lexicon, tokenDict);
    }
    // create an empty noiselm
    noiselm = std::make_shared<NoiseLMLetterSwapUnit>(
        "",
        "identitynoiselm",
        noise_keys,
        FLAGS_allowSwap,
        FLAGS_allowInsertion,
        FLAGS_allowDeletion,
        false,
        FLAGS_scale_noise,
        1,
        1,
        1,
        0);
  }
  dicts.insert({kCleanKeyIdx, tokenDict});
  dicts.insert({kNoiseKeyIdx, tokenDict});

  // Dictionary lmDict = createFairseqTokenDict(FLAGS_lmdict);

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;

  /* =========== ASG BEAM NOISE SPECIFIC ============ */
  std::shared_ptr<AutoSegBeamNoiseCriterion> asgbeamnoisecrit;
  std::shared_ptr<fl::FirstOrderOptimizer> scaleoptim;
  /* ================================================ */
  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  if (runStatus == kTrainMode) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();

    network = createW2lSeqModule(archfile, numFeatures, numClasses);

    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqCriterion) {
      criterion = std::make_shared<Seq2SeqCriterion>(
          buildSeq2Seq(numClasses, tokenDict.getIndex(kEosToken)));
    }
    /* =========== ASG BEAM NOISE SPECIFIC ============ */
    else if (FLAGS_criterion == kAsgBeamNoiseCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(
          numClasses,
          tokenDict,
          noiselex,
          *noiselm,
          FLAGS_beamsize,
          scalemode,
          FLAGS_beamthreshold,
          FLAGS_computeStats,
          FLAGS_topk,
          FLAGS_useevalemission,
          FLAGS_useNoiseToSort);
    }
    /* ================================================ */

    else if (FLAGS_criterion == kTransformerCriterion) {
      criterion =
          std::make_shared<TransformerCriterion>(buildTransformerCriterion(
              numClasses,
              FLAGS_am_decoder_tr_layers,
              FLAGS_am_decoder_tr_dropout,
              FLAGS_am_decoder_tr_layerdrop,
              tokenDict.getIndex(kEosToken)));
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }

    // lmcrit = createLMCritic(lmDict, dict);
  } else if (runStatus == kForkMode) {
    std::unordered_map<std::string, std::string> cfg; // unused
    /* =========== ASG BEAM NOISE SPECIFIC ============ */
    if (noiselm && wasASGBeamNoise) {
      std::vector<fl::Variable> noiselmparams;
      // netoptim and critoptim loaded for nothing
      W2lSerializer::load(
          reloadPath,
          cfg,
          network,
          criterion,
          netoptim,
          critoptim,
          noiselmparams);
      for (int64_t i = 0; i < noiselmparams.size(); i++) {
        noiselm->params().at(i).array() = noiselmparams.at(i).array();
      }
      noiselm->paramsToCpu();
      isNoiseModelTrained = true;
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(
          numClasses,
          tokenDict,
          noiselex,
          *noiselm,
          FLAGS_beamsize,
          scalemode,
          FLAGS_beamthreshold,
          FLAGS_computeStats,
          FLAGS_topk,
          FLAGS_useevalemission,
          FLAGS_useNoiseToSort);
    } else {
      if (noiselm) {
        asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(
            numClasses,
            tokenDict,
            noiselex,
            *noiselm,
            FLAGS_beamsize,
            scalemode,
            FLAGS_beamthreshold,
            FLAGS_computeStats,
            FLAGS_topk,
            FLAGS_useevalemission,
            FLAGS_useNoiseToSort);
      }
      /* ================================================ */
      W2lSerializer::load(reloadPath, cfg, network, criterion);
    }

  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused

    /* =========== ASG BEAM NOISE SPECIFIC ============ */

    if (noiselm) {
      std::vector<fl::Variable> noiselmparams;
      W2lSerializer::load(
          reloadPath,
          cfg,
          network,
          criterion,
          netoptim,
          critoptim,
          noiselmparams,
          scaleoptim);
      if (noiselm) {
        for (int64_t i = 0; i < noiselmparams.size(); i++) {
          noiselm->params().at(i).array() = noiselmparams.at(i).array();
        }
        noiselm->paramsToCpu();
        isNoiseModelTrained = true;
      }
      scaleoptim = initParamOptimizer(
          {noiselm->params()[0]},
          FLAGS_scaleoptim,
          scaleoptim->getLr(),
          0.0,
          0.0);
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(
          numClasses,
          tokenDict,
          noiselex,
          *noiselm,
          FLAGS_beamsize,
          scalemode,
          FLAGS_beamthreshold,
          FLAGS_computeStats,
          FLAGS_topk,
          FLAGS_useevalemission,
          FLAGS_useNoiseToSort);
    } else {
      // std::cout << "load in continue mode" << std::endl;
      W2lSerializer::load(
          reloadPath, cfg, network, criterion, netoptim, critoptim);
      // W2lSerializer::load(reloadPath, cfg, network, criterion);
      // netoptim = initOptimizer(
      //  {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum,
      //  FLAGS_weightdecay);
      // critoptim =
      //  initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
    }
  }

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);

    /* =========== ASG BEAM NOISE SPECIFIC ============ */
    if (noiselm) {
      scaleoptim = initParamOptimizer(
          {noiselm->params()[0]},
          FLAGS_scaleoptim,
          FLAGS_lrscalenoise,
          0.0,
          0.0);
    }
    /* ================================================ */
  }

  // needed to find a good scale
  auto falcrit =
      std::make_shared<ForceAlignmentCriterion>(numClasses, scalemode);

  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();
  LOG_MASTER(INFO) << "[Criterion Params: " << numTotalParams(criterion) << "]";
  if (noiselm) {
    LOG_MASTER(INFO) << "[Noiselm] " << noiselm->prettyString();
    LOG_MASTER(INFO) << "[Noiselm Params: " << numTotalParams(noiselm) << "]";
  }
  LOG_MASTER(INFO) << "[Optimizer network] " << netoptim->prettyString();
  LOG_MASTER(INFO) << "[Optimizer criterion] " << critoptim->prettyString();
  if (noiselm) {
    LOG_MASTER(INFO) << "[Optimizer scale] " << scaleoptim->prettyString();
  }
  /* ===================== Create Dataset ===================== */
  auto pairedDs = createDatasetNoise(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);

  std::shared_ptr<NoiseW2lListFilesDataset> pairednoiseDs;
  if (!FLAGS_trainnoise.empty()) {
    pairednoiseDs = createDatasetNoise(
        FLAGS_trainnoise,
        dicts,
        lexicon,
        FLAGS_batchsize,
        worldRank,
        worldSize);
  }

  std::shared_ptr<NoiseW2lListFilesDataset> unpairedAudioDs;
  if (!FLAGS_trainaudio.empty()) {
    unpairedAudioDs = createDatasetNoise(
        FLAGS_trainaudio,
        dicts,
        lexicon,
        FLAGS_audiobatchsize == 0 ? FLAGS_batchsize : FLAGS_audiobatchsize,
        worldRank,
        worldSize);
    // eraseTargets of the dataset
    for (int64_t idx = 0; idx < unpairedAudioDs->size(); idx++) {
      unpairedAudioDs->eraseTargets(idx);
    }
  }

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset(s)";
    pairedDs->shuffle(FLAGS_seed);
    if (!FLAGS_trainaudio.empty()) {
      unpairedAudioDs->shuffle(FLAGS_seed);
    }
    if (!FLAGS_trainnoise.empty()) {
      pairednoiseDs->shuffle(FLAGS_seed);
    }
  }

  /* ===================== Hooks ===================== */
  if (reducer) {
    fl::distributeModuleGrads(network, reducer);
    fl::distributeModuleGrads(criterion, reducer);
    // if (noiselm){
    //  fl::distributeModuleGrads(noiselm, reducer);
    //}
  }

  fl::allReduceParameters(network);
  fl::allReduceParameters(criterion);

  bool isPairedData;
  network->train();
  criterion->train();

  double initlr = netoptim->getLr();
  double initcritlr = critoptim->getLr();
  // double initnoiselmlr;
  // scaleoptim->setLr(initnoiselmlr);

  af::sync();

  network->eval();

  for (int64_t idx = 0; idx < unpairedAudioDs->size(); idx++) {
    auto unp_sample = unpairedAudioDs->get(idx);
    auto output_eval =
        network->forward({fl::input(unp_sample[kInputIdx])}).front();
    auto newTranscriptions =
        getUpdateTrancriptsWords(output_eval, criterion, dicts);
    unpairedAudioDs->updateTargets(idx, newTranscriptions);
  }

  network->train();
  af::sync();

  // Noise model traning loop
  // We can update the noise model on paired data
  /*
  auto statsUnpaired = w2l::SpeechStatMeter();
  noiselm->trainModel(
          pairednoiseDs,
          network,
          criterion->param(0),
          dicts,
          FLAGS_enable_distributed,
          (int)FLAGS_replabel,
          statsUnpaired

  );
  */
  if (isMaster)
    noiselm->displayNoiseModel();

  auto updateScale = [&falcrit,
                      &criterion,
                      &asgbeamnoisecrit,
                      &unpairedAudioDs,
                      &network,
                      &noiselm,
                      &scaleoptim,
                      &reducer,
                      &isMaster](int nbIter) {
    auto output_eval = fl::Variable();
    falcrit->setParams(criterion->param(0), 0);
    // network->eval();
    fl::Variable fal, fal_beam, lossScale;
    int iterScale = 0;
    while (iterScale < nbIter) {
      for (auto& sample : *unpairedAudioDs) {
        iterScale++;
        auto input = fl::input(sample[kInputIdx]);
        auto output = network->forward({input}).front();

        auto resforward = asgbeamnoisecrit->forward(
            output,
            output_eval,
            criterion->param(0),
            fl::noGrad(sample[kTargetIdx]),
            fl::noGrad(sample[kNoiseKeyIdx]));
        fal_beam = resforward[2];

        fal = falcrit->forward(output, fl::noGrad(sample[kTargetIdx]));
        lossScale = fal - fal_beam;
        af::print("fal", fal.array());
        af::print("fal_beam", fal_beam.array());
        // meters.train.losses[klossScale].add(lossScale.array());
        lossScale = fl::sum(lossScale, {0});
        if (reducer)
          fl::allReduce(lossScale);
        af::sync();

        double lossScale_cpu = lossScale.scalar<float>();
        double grad_cpu = lossScale_cpu > 0 ? 1 : (lossScale_cpu < 0 ? -1 : 0);
        noiselm->params()[0].addGrad(
            fl::Variable(af::array(1, &grad_cpu), false));
        scaleoptim->step();
        scaleoptim->zeroGrad();
        noiselm->scaleToCpu();
        if (isMaster)
          af::print("scale af", noiselm->params()[0].array());
        if (iterScale >= nbIter) {
          break;
        }
      }
    }
  };

  updateScale(FLAGS_iterscale);

  return 0;
}
