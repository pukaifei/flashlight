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

#include "experimental/lead2Gold/src/common/Defines.h"
#include "common/FlashlightUtils.h"
#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "experimental/lead2Gold/src/data/Featurize.h"
#include "experimental/lead2Gold/src/data/Utils.h"
#include "experimental/lead2Gold/src/common/Utils.h"
#include "experimental/lead2Gold/src/runtime/runtime.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "common/Transforms.h"


using namespace w2l;

int main(int argc, char** argv) {
  FLAGS_logtostderr = 0;
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 0;
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
  bool wasCTCBeamNoise = config["wasCTCBeamNoise"] == "1";
  int runIdx = std::stoi(config[kRunIdx]);
  std::string reloadPath = config[kReloadPath];
  int startEpoch = std::stoi(config[kStartEpoch]);
  int startIter = std::stoi(config[kStartIter]);
  std::string runPath = config[kRunPath];
  std::string runStatus = config[kRunStatus];
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
  if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kCtcBeamNoiseCriterion) {
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
  //to add LM look at wave2word code
  std::shared_ptr<NoiseTrie> noiselex = nullptr;
  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm;
  bool isNoiseModelTrained = false;
  Dictionary noise_keys(dictPath);
  if(FLAGS_criterion == kAsgBeamNoiseCriterion || FLAGS_criterion == kCtcBeamNoiseCriterion) {
    if (FLAGS_uselexicon && !FLAGS_lexicon.empty()) {
      noiselex = std::shared_ptr<NoiseTrie>(new NoiseTrie(noise_keys.indexSize(), noise_keys.getIndex("|"), nullptr));
      auto words = noiselex->load(FLAGS_lexicon, tokenDict);
    }
    //create an empty noiselm
    noiselm = std::make_shared<NoiseLMLetterSwapUnit>(
      "", "identitynoiselm", noise_keys, FLAGS_allowSwap,
      FLAGS_allowInsertion, FLAGS_allowDeletion,
      false, FLAGS_scale_noise, FLAGS_scale_sub, FLAGS_scale_ins, FLAGS_scale_del, 0);
  }
  dicts.insert({kCleanKeyIdx, tokenDict});
  dicts.insert({kNoiseKeyIdx, tokenDict});
  
  //Dictionary lmDict = createFairseqTokenDict(FLAGS_lmdict);

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;

  /* =========== ASG BEAM NOISE SPECIFIC ============ */
  std::shared_ptr<AutoSegBeamNoiseCriterion> asgbeamnoisecrit;
  std::shared_ptr<CtcBeamNoiseCriterion> ctcbeamnoisecrit;
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
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
    }
    else if (FLAGS_criterion == kCtcBeamNoiseCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
      ctcbeamnoisecrit = std::make_shared<CtcBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort, FLAGS_nbNested);
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

    //lmcrit = createLMCritic(lmDict, dict);
  } else if (runStatus == kForkMode) {
    std::unordered_map<std::string, std::string> cfg; // unused
    /* =========== ASG BEAM NOISE SPECIFIC ============ */
    if (noiselm && wasASGBeamNoise) {
      std::vector<fl::Variable> noiselmparams;
      //netoptim and critoptim loaded for nothing
      W2lSerializer::load(reloadPath, cfg, network, criterion, netoptim, critoptim, noiselmparams);
      for(int64_t i = 0; i < noiselmparams.size(); i++) {
        noiselm->params().at(i).array() = noiselmparams.at(i).array();
      }
      noiselm->paramsToCpu();
      isNoiseModelTrained = true;
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
    } else if (noiselm && wasCTCBeamNoise){
      std::vector<fl::Variable> noiselmparams;
      //netoptim and critoptim loaded for nothing
      W2lSerializer::load(reloadPath, cfg, network, criterion, netoptim, critoptim, noiselmparams);
      for(int64_t i = 0; i < noiselmparams.size(); i++) {
        noiselm->params().at(i).array() = noiselmparams.at(i).array();
      }
      noiselm->paramsToCpu();
      isNoiseModelTrained = true;
      ctcbeamnoisecrit = std::make_shared<CtcBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);

    } else {

    if ( FLAGS_criterion == kAsgBeamNoiseCriterion ) {
      asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
    }
    if ( FLAGS_criterion == kCtcBeamNoiseCriterion ) {
      ctcbeamnoisecrit = std::make_shared<CtcBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
    }
    /* ================================================ */
      W2lSerializer::load(
        reloadPath, cfg, network, criterion);
    }

  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused

    /* =========== ASG BEAM NOISE SPECIFIC ============ */

    if (noiselm){
      std::vector<fl::Variable> noiselmparams;
      W2lSerializer::load(reloadPath, cfg, network, criterion, netoptim, critoptim, noiselmparams, scaleoptim);
      if (noiselm){
        for(int64_t i = 0; i < noiselmparams.size(); i++) {
          noiselm->params().at(i).array() = noiselmparams.at(i).array();
        }
        noiselm->paramsToCpu();
        noiselm->displayNoiseModel();
        isNoiseModelTrained = true;

      }
      scaleoptim = initParamOptimizer({noiselm->params()[0]}, FLAGS_scaleoptim, scaleoptim->getLr() , 0.0, 0.0);
      if ( FLAGS_criterion == kAsgBeamNoiseCriterion ) {
        asgbeamnoisecrit = std::make_shared<AutoSegBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
      }
      if ( FLAGS_criterion == kCtcBeamNoiseCriterion ) {
        ctcbeamnoisecrit = std::make_shared<CtcBeamNoiseCriterion>(numClasses, tokenDict, noiselex, *noiselm, FLAGS_beamsize, scalemode, FLAGS_beamthreshold, FLAGS_computeStats, FLAGS_topk, FLAGS_useevalemission, FLAGS_useNoiseToSort);
      }
    } else{
      //std::cout << "load in continue mode" << std::endl;
      W2lSerializer::load(reloadPath, cfg, network, criterion, netoptim, critoptim);
      //W2lSerializer::load(reloadPath, cfg, network, criterion);
      //netoptim = initOptimizer(
      //  {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
      //critoptim =
      //  initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  
    }

  }

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  
    /* =========== ASG BEAM NOISE SPECIFIC ============ */
    if (noiselm){
      scaleoptim = initParamOptimizer({noiselm->params()[0]}, FLAGS_scaleoptim, FLAGS_lrscalenoise, 0.0, 0.0);
    }
    /* ================================================ */
  }

  //needed to find a good scale
  auto falcrit = std::make_shared<ForceAlignmentCriterion>(numClasses, scalemode);
  auto ctccrit = std::make_shared<CTCLoss>(scalemode);

  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();
  LOG_MASTER(INFO) << "[Criterion Params: " << numTotalParams(criterion) << "]";
  if (noiselm){
    LOG_MASTER(INFO) << "[Noiselm] " << noiselm->prettyString();
    LOG_MASTER(INFO) << "[Noiselm Params: " << numTotalParams(noiselm) << "]";
  }
  LOG_MASTER(INFO) << "[Optimizer network] " << netoptim->prettyString();
  LOG_MASTER(INFO) << "[Optimizer criterion] " << critoptim->prettyString();
  if (noiselm){
    LOG_MASTER(INFO) << "[Optimizer scale] " << scaleoptim->prettyString();
  }
  /* ===================== Create Dataset ===================== */
  auto pairedDs = createDatasetNoise(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);


  std::shared_ptr<NoiseW2lListFilesDataset> pairednoiseDs;
  if (!FLAGS_trainnoise.empty()){
    pairednoiseDs = createDatasetNoise(
        FLAGS_trainnoise, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  }


  std::shared_ptr<NoiseW2lListFilesDataset> unpairedAudioDs;
  bool noTarget;
  if (!FLAGS_trainaudio.empty()){
    unpairedAudioDs = createDatasetNoise(
        FLAGS_trainaudio,
        dicts,
        lexicon,
        FLAGS_audiobatchsize == 0 ? FLAGS_batchsize : FLAGS_audiobatchsize,
        worldRank,
        worldSize);
    //eraseTargets of the dataset
    for (int64_t idx=0 ; idx < unpairedAudioDs->size() ; idx++){
      unpairedAudioDs->eraseTargets(idx);
    }
    noTarget=true;
  }

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset(s)";
    pairedDs->shuffle(FLAGS_seed);
    if (!FLAGS_trainaudio.empty()){
      unpairedAudioDs->shuffle(FLAGS_seed);
    }
    if (!FLAGS_trainnoise.empty()){
      pairednoiseDs->shuffle(FLAGS_seed);
    }
  }

  auto trainEvalIds =
      getTrainEvalIds(pairedDs->size(), FLAGS_pcttraineval, FLAGS_seed);

  auto validSets = split(',', trim(FLAGS_valid));
  std::unordered_map<std::string, std::shared_ptr<W2lDataset>> validds;
  for (const auto& s : validSets) {
    auto ts = splitOnAnyOf(":", s);
    auto setKey = ts.size() == 1 ? s : ts[0];
    auto setValue = ts.size() == 1 ? s : ts[1];

    validds[setKey] = createDatasetNoise(
        setValue, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  }

  /* ===================== Training Dataset Scheduler ===================== */
  if (FLAGS_pairediter == 0){
    FLAGS_pairediter = pairedDs->size();
  }
  if (FLAGS_trainaudio.empty()){
    FLAGS_audioiter = 0;
  } else if (FLAGS_audioiter == 0){
    if (FLAGS_ratioaudio == 0){
      FLAGS_audioiter = unpairedAudioDs->size();
    } else{
      FLAGS_audioiter = std::min((int)FLAGS_ratioaudio * FLAGS_pairediter, unpairedAudioDs->size());
    }
  }

  std::vector<std::shared_ptr<W2lDataset>> param_schedule_ds;
  std::vector<int64_t> param_schedule_datatypes;
  std::vector<int64_t> param_schedule_numiter;
  if (!FLAGS_trainaudio.empty()){
    param_schedule_ds = {pairedDs, unpairedAudioDs};
    param_schedule_datatypes = {kParallelData, kUnpairedAudio};
    param_schedule_numiter = {FLAGS_pairediter, FLAGS_audioiter};
  } else{
    param_schedule_ds = {pairedDs};
    param_schedule_datatypes = {kParallelData};
    param_schedule_numiter = {FLAGS_pairediter};
  }

  DataScheduler trainDscheduler(
      param_schedule_ds,
      param_schedule_datatypes,
      param_schedule_numiter,
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
    //if (noiselm){
    //  fl::distributeModuleGrads(noiselm, reducer);
    //}
  }

  fl::allReduceParameters(network);
  fl::allReduceParameters(criterion);
  //if (noiselm){
  //  fl::allReduceParameters(noiselm);
  //}
  
  auto updateScaleAsg = [&falcrit,
                    &criterion,
                    &asgbeamnoisecrit,
                    &unpairedAudioDs,
                    &network,
                    &noiselm,
                    &scaleoptim,
                    &reducer,
                    &isMaster,
                    &meters](int nbIter) {

    auto output_eval = fl::Variable();
    falcrit->setParams(criterion->param(0), 0);
    //network->eval();
    fl::Variable fal, fal_beam, lossScale;
    int iterScale=0;
    while (iterScale < nbIter){
      for (auto& sample : *unpairedAudioDs){
        iterScale++;
        auto input = fl::input(sample[kInputIdx]);
        auto output = network->forward({input}).front();

        fal_beam = asgbeamnoisecrit->forward(output, output_eval, criterion->param(0), fl::noGrad(sample[kTargetIdx]), fl::noGrad(sample[kNoiseKeyIdx]))[2];
        //fal_beam = resforward[2];

        fal = falcrit->forward(output, fl::noGrad(sample[kTargetIdx]));
        lossScale = fal - fal_beam;
        meters.noiselm.losses[klossScale].add(lossScale.array());
        lossScale = fl::sum(lossScale, {0});
        if (reducer)
          fl::allReduce(lossScale);
        af::sync();

        double lossScale_cpu = lossScale.scalar<float>();
        double grad_cpu = lossScale_cpu > 0 ? 1 : (lossScale_cpu < 0 ? -1 : 0);
        noiselm->params()[0].addGrad(fl::Variable(af::array(1, &grad_cpu), false));
        scaleoptim->step();
        scaleoptim->zeroGrad();
        noiselm->scaleToCpu();
        if (iterScale >= nbIter){
          break;
        }
      }
    }
  };

  auto updateScaleCtc = [&ctccrit,
                    &ctcbeamnoisecrit,
                    &unpairedAudioDs,
                    &network,
                    &noiselm,
                    &scaleoptim,
                    &reducer,
                    &isMaster,
                    &meters](int nbIter) {

    auto output_eval = fl::Variable();
    //network->eval();
    fl::Variable fal, fal_beam, lossScale;
    int iterScale=0;
    while (iterScale < nbIter){
      for (auto& sample : *unpairedAudioDs){
        iterScale++;
        auto input = fl::input(sample[kInputIdx]);
        auto output = network->forward({input}).front();

        fal_beam = ctcbeamnoisecrit->forward(output, output_eval, fl::noGrad(sample[kTargetIdx]))[0];
        fal = ctccrit->forward({output, fl::noGrad(sample[kTargetIdx])})[0];
        lossScale = fal - fal_beam;
        meters.noiselm.losses[klossScale].add(lossScale.array());
        lossScale = fl::sum(lossScale, {0});
        if (reducer)
          fl::allReduce(lossScale);
        af::sync();

        double lossScale_cpu = lossScale.scalar<float>();
        double grad_cpu = lossScale_cpu > 0 ? 1 : (lossScale_cpu < 0 ? -1 : 0);
        noiselm->params()[0].addGrad(fl::Variable(af::array(1, &grad_cpu), false));
        scaleoptim->step();
        scaleoptim->zeroGrad();
        noiselm->scaleToCpu();
        if (iterScale >= nbIter){
          break;
        }
      }
    }
  };

  auto train = [&noTarget,
                &meters,
                &trainEvalIds,
                &unpairedAudioDs,
                &trainDscheduler,
                &pairednoiseDs,
                &validds,
                &startEpoch,
                &startIter,
                &nItersPerEpoch,
                &network,
                &criterion,
                &asgbeamnoisecrit,
                &ctcbeamnoisecrit,
                &noiselm,
                &isNoiseModelTrained,
                &falcrit,
                &netoptim,
                &critoptim,
                &scaleoptim,
                &dicts,
                &config,
                &logHelper,
                &logOnEpoch,
                &reducer,
                &isMaster,
                &updateScaleAsg,
                &updateScaleCtc](int nEpochs) {
    int64_t curEpoch = startEpoch;
    int64_t curIter = startIter;
    bool isPairedData;
    network->train();
    criterion->train();
    //lmcrit->eval();
    
    double initlr = netoptim->getLr();
    double initcritlr = critoptim->getLr();
    //double initnoiselmlr;
    //if (noiselm){
    //  initnoiselmlr = scaleoptim->getLr();
    //}
    
    std::shared_ptr<SpecAugment> saug;
    if (FLAGS_saug_start_update >= 0) {
      saug = std::make_shared<SpecAugment>(
          FLAGS_filterbanks,
          FLAGS_saug_fmaskf,
          FLAGS_saug_fmaskn,
          FLAGS_saug_tmaskt,
          FLAGS_saug_tmaskp,
          FLAGS_saug_tmaskn);
    }

    while (curEpoch < nEpochs) {

      //double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
      //netoptim->setLr(lrScale * FLAGS_lr);

      ++curEpoch;
      if (curEpoch >= FLAGS_lr_decay &&
          (curEpoch - FLAGS_lr_decay) % FLAGS_lr_decay_step == 0) {
        initlr /= 2;
        initcritlr /= 2;
        //if (noiselm){
        //initnoiselmlr /= 2;
        //scaleoptim->setLr(initnoiselmlr);
        //}
      }

      // linearly warm up the amount of unpaired audio data used in training
      if (FLAGS_audiowarmupepochs > 0 && curEpoch > FLAGS_pretrainWindow &&
          (curEpoch - FLAGS_pretrainWindow) <= FLAGS_audiowarmupepochs) {
        int unpairedIter = (curEpoch - FLAGS_pretrainWindow) * FLAGS_audioiter /
            FLAGS_audiowarmupepochs;
        trainDscheduler.setSchedule({FLAGS_pairediter, unpairedIter});
        nItersPerEpoch = FLAGS_pairediter + unpairedIter;
      }

      af::sync();
      
      meters.timer[kRuntime].resume();
      
      LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";

      if (unpairedAudioDs){
        if (!FLAGS_updateOnTheFly &&
                  ( (curEpoch-1) % FLAGS_updateTranscriptEveryNEpoch == 0)
                    || noTarget){
          meters.timer[kUpdateTransTimer].resume();
          network->eval();

          for (int64_t idx=0 ; idx < unpairedAudioDs->size() ; idx++){
            auto unp_sample = unpairedAudioDs->get(idx);
            auto output_eval = network->forward({fl::input(unp_sample[kInputIdx])}).front();
            auto newTranscriptions = getUpdateTrancriptsWords(output_eval, criterion, dicts);
            unpairedAudioDs->updateTargets(idx, newTranscriptions);
          }
          noTarget=false;

          network->train();
          af::sync();
          meters.timer[kUpdateTransTimer].stopAndIncUnit();
        }
      }


      //Noise model traning loop
      //We can update the noise model on paired data
      resetNoiselmMeters(meters.noiselm);
      meters.timer[kUpdateNoiseModelTimer].resume();
      if (noiselm && (FLAGS_identityTest == false) && ((FLAGS_updatedNoiseModelEveryNEpoch != 0 && (curEpoch-1) % FLAGS_updatedNoiseModelEveryNEpoch == 0) || isNoiseModelTrained == false)) {
        noiselm->trainModel(
            pairednoiseDs,
            network,
            criterion,
            dicts,
            FLAGS_enable_distributed,
            (int)FLAGS_replabel,
            meters.statsNoise
        );
        if (isMaster)
          noiselm->displayNoiseModel();
        if (FLAGS_evalnoiselm){
          noiselm->evalModel(
            network,
            criterion,
            (int)FLAGS_replabel,
            validds["dev-clean"],
            dicts,
            meters.noiselm.losses[klossNoiselm]
          );
        }
        isNoiseModelTrained = true;
      }
      meters.timer[kUpdateNoiseModelTimer].stopAndIncUnit();

      
      if (noiselm && (FLAGS_updateScaleEveryNEpoch != 0 && (curEpoch-1) % FLAGS_updateScaleEveryNEpoch == 0)){
        meters.timer[kUpdateScaleTimer].resume();
        if (asgbeamnoisecrit){
          if (curEpoch-1 == 0){
            updateScaleAsg(FLAGS_iterscale*5); // initial training has to be longer
          } else{
            updateScaleAsg(FLAGS_iterscale);
          }
        } else if (ctcbeamnoisecrit){
          if (curEpoch-1 == 0){
            updateScaleCtc(FLAGS_iterscale*5); // initial training has to be longer
          } else{
            updateScaleCtc(FLAGS_iterscale);
          }
        }
        meters.timer[kUpdateScaleTimer].stopAndIncUnit();    
      }
      

      int scheduleIter = 0;
      //ASR training loop
      meters.timer[kSampleTimer].resume();
      meters.timer[kTimer].resume();
      while (scheduleIter < nItersPerEpoch) {
        auto sample = trainDscheduler.get();
        isPairedData = af::allTrue<bool>(sample[kDataTypeIdx] == kParallelData);
        ++curIter;
        ++scheduleIter;

        double lrScale = 1;
        if (FLAGS_lrcosine) {
          const double pi = std::acos(-1);
          lrScale =
              std::cos(((double)curIter) / ((double)nEpochs*nItersPerEpoch) * pi / 2.0);
        } else {
          lrScale =
              std::pow(FLAGS_gamma, (double)curIter / (double)FLAGS_stepsize);
        }

        netoptim->setLr(
            lrScale * initlr * std::min(curIter / double(FLAGS_warmup), 1.0));
        critoptim->setLr(
            lrScale * initcritlr * std::min(curIter / double(FLAGS_warmup), 1.0));



        af::sync();

        meters.timer[kTimer].incUnit();
        meters.timer[kSampleTimer].stopAndIncUnit();
        if (isPairedData)
          meters.stats.add(sample[kInputIdx], sample[kTargetIdx]);
        else
          meters.statsUnpaired.add(sample[kInputIdx], af::array());
        if (af::anyTrue<bool>(af::isNaN(sample[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(sample[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values";
        }

        auto input = fl::input(sample[kInputIdx]);
        //optionally compute outpute in eval mode to update transcriptions
        //We have to do it before because .eval() erases gradients.
        fl::Variable output_eval = fl::Variable();
        //meters.timer[kUpdateTransTimer].resume();
        meters.timer[kFwdTimer].resume();
        if (!isPairedData && 
              ( (FLAGS_updateOnTheFly && (curEpoch-1) % FLAGS_updateTranscriptEveryNEpoch == 0) ||
              af::anyTrue<bool>(sample[kNoiseKeyIdx].isempty()) ||
              FLAGS_useevalemission
              )
            ) {
          network->eval();
          output_eval = network->forward({input}).front();
          network->train();  
        }
        //meters.timer[kUpdateTransTimer].stopAndIncUnit();

        // forward
        if (FLAGS_saug_start_update >= 0 &&
            curIter >= FLAGS_saug_start_update) {
          input = saug->forward(input);
        }
        auto output = network->forward({input}).front();
        af::sync();
        meters.timer[kFwdTimer].stopAndIncUnit();
        fl::Variable loss;
        if (isPairedData) {
          //std::cout << "Paired data" << std::endl;
          meters.timer[kCritFwdTimer].resume();
          loss = criterion->forward({output, fl::noGrad(sample[kTargetIdx])})[0];
          af::sync();
          meters.timer[kCritFwdTimer].stopAndIncUnit();
          if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
            LOG(FATAL) << "ASR paired loss has NaN values";
          }
          meters.train.losses[kASRPaired].add(loss.array());
        } else {
          //std::cout << "Unpaired data" << std::endl;

          //Optionally update the noisy transcription
          meters.timer[kUpdateTransTimer].resume();
          if ( (FLAGS_updateOnTheFly && (curEpoch-1) % FLAGS_updateTranscriptEveryNEpoch == 0) ||
              af::anyTrue<bool>(sample[kNoiseKeyIdx].isempty())){
            if (output_eval.isempty()){
              LOG(FATAL) << "Output eval is empty";
            }
            updateTrancripts(sample, output_eval, criterion, dicts);
          }
          af::sync();
          meters.timer[kUpdateTransTimer].stopAndIncUnit();

          meters.timer[kCritFwdNoiseTimer].resume();

          if (noiselm){
            if (asgbeamnoisecrit){
              loss = asgbeamnoisecrit->forward(output, output_eval, criterion->param(0), fl::noGrad(sample[kTargetIdx]), fl::noGrad(sample[kNoiseKeyIdx]))[0];
            } else if (ctcbeamnoisecrit){
              loss = ctcbeamnoisecrit->forward(output, output_eval, fl::noGrad(sample[kTargetIdx]))[0];
            }

          } else{
            loss = criterion->forward({output, fl::noGrad(sample[kTargetIdx])})[0];
          }
          af::sync();
          meters.timer[kCritFwdNoiseTimer].stopAndIncUnit();
          meters.train.losses[kASRUnpaired].add(loss.array());
        }


        // compute training error rate from parallel data
        if (isPairedData) {
          evalOutput(
              output.array(),
              sample[kTargetIdx],
              meters.train.edits,
              dicts[kTargetIdx],
              criterion);
        }
        // backward
        meters.timer[kBwdTimer].resume();
        netoptim->zeroGrad();
        critoptim->zeroGrad();

        loss.backward();
        if (reducer) {
          reducer->finalize();
        }

        af::sync();
        meters.timer[kBwdTimer].stopAndIncUnit();
        meters.timer[kOptimTimer].resume();
        // scale down gradients by batchsize
        for (const auto& p : network->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / sample[kInputIdx].dims(3);
        }
        for (const auto& p : criterion->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / sample[kInputIdx].dims(3);
        }

        if (FLAGS_maxgradnorm > 0) {
          auto params = network->params();
          auto critparams = criterion->params();
          params.insert(params.end(), critparams.begin(), critparams.end());
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }
        netoptim->step();
        critoptim->step();
        af::sync();
        meters.timer[kOptimTimer].stopAndIncUnit();
        meters.timer[kSampleTimer].resume();

        // checkpoint evaluation

        if ((!logOnEpoch && curIter % FLAGS_reportiters == 0) ||
            (logOnEpoch && scheduleIter == nItersPerEpoch)) {
          stopTimeMeters(meters);
          runEval(
              network, criterion, asgbeamnoisecrit, noiselm, FLAGS_replabel, validds, meters, dicts, FLAGS_evalbeamnoise);
          config[kEpoch] = std::to_string(curEpoch);
          config[kIteration] = std::to_string(curIter);
          //std::unordered_map<std::string, double> logFields(
          //    {{"lr", netoptim->getLr()},
          //     {"lmcrit-t", lmTempScale * FLAGS_gumbeltemperature}});

          std::unordered_map<std::string, double> logFields(
              {{"lr-net", netoptim->getLr()},
              {"lr-crit", critoptim->getLr()}}); // add noise model stats
          logFields.insert({"lr-sc", noiselm ? scaleoptim->getLr() : 0});
          logFields.insert({"sc-noise", noiselm ? noiselm->scale_noise() : 0});

          logHelper.logAndSaveModel(
              meters, config, network, criterion, netoptim, critoptim, noiselm, scaleoptim, logFields);
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

  //Create a file to indicate that training is finished. 
  //Slurm will know that no reschedule is needed.
  if (isMaster){
    try {
      auto filename = getRunFile("finished", 0, runPath);
      std::ofstream output(filename);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error while writing finish file: " << ex.what();
    }
  }

  return 0;
}
