/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/lead2Gold/src/runtime/Logger.h"

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thread>

#include "common/FlashlightUtils.h"
#include "experimental/lead2Gold/src/common/Defines.h"
#include "runtime/Serial.h"

namespace w2l {

// For Train.cpp
std::pair<std::string, std::string> getStatus(
    NoiseTrainMeters& meters,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    bool verbose /* = false */,
    bool date /* = false */,
    const std::string& separator /* = " " */) {
  std::string errtype = "XER";
  errtype[0] = std::toupper(FLAGS_target[0]);
  std::string header, status;
  auto insertItem = [&](std::string key, std::string val) {
    if (verbose) {
      val = key + ": " + val;
    }
    header = header + (header.empty() ? "" : separator) + key;
    status = status + (status.empty() ? "" : separator) + val;
  };
  if (date) {
    insertItem("date", format("%s", getCurrentDate().c_str()));
    insertItem("time", format("%s", getCurrentTime().c_str()));
  }
  insertItem("epoch", format("%8d", epoch));
  insertItem("nupdates", format("%12d", nupdates));
  insertItem("lr", format("%4.6lf", lr));
  insertItem("lrcriterion", format("%4.6lf", lrcrit));

  int rt = meters.runtime.value();
  insertItem(
      "runtime",
      format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60));
  insertItem("bch(ms)", format("%.2f", meters.timer.value() * 1000));
  insertItem("smp(ms)", format("%.2f", meters.sampletimer.value() * 1000));
  insertItem("fwd(ms)", format("%.2f", meters.fwdtimer.value() * 1000));
  insertItem(
      "crit-fwd(ms)", format("%.2f", meters.critfwdtimer.value() * 1000));
  insertItem("bwd(ms)", format("%.2f", meters.bwdtimer.value() * 1000));
  insertItem("optim(ms)", format("%.2f", meters.optimtimer.value() * 1000));
  insertItem("loss", format("%10.5f", meters.train.loss.value()[0]));

  insertItem(
      "train-" + errtype, format("%5.2f", meters.train.tknEdit.value()[0]));
  insertItem("train-WER", format("%5.2f", meters.train.wrdEdit.value()[0]));
  for (auto& v : meters.valid) {
    insertItem(v.first + "-loss", format("%10.5f", v.second.loss.value()[0]));
    insertItem(
        v.first + "-" + errtype, format("%5.2f", v.second.tknEdit.value()[0]));
    insertItem(v.first + "-WER", format("%5.2f", v.second.wrdEdit.value()[0]));
  }
  auto stats = meters.stats.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem("avg-isz", format("%03d", isztotal / numsamples));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec = audioProcSec * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = meters.timer.value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");
  insertItem("train-wLER", format("%5.2f", meters.train.wLER.value()[0]));
  return {header, status};
}

LogHelper::LogHelper(
    int runIdx,
    std::string runPath,
    bool isMaster,
    bool logOnEpoch)
    : runIdx_(runIdx),
      runPath_(runPath),
      isMaster_(isMaster),
      logOnEpoch_(logOnEpoch) {
  if (isMaster_) {
    logFileName_ = getRunFile("log", runIdx_, runPath_);
    perfFileName_ = getRunFile("perf", runIdx_, runPath_);
    dirCreate(runPath_);
    std::ofstream logFile, perfFile;
    logFile.open(logFileName_);
    if (!logFile.is_open()) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    perfFile.open(perfFileName_);
    if (!perfFile.is_open()) {
      LOG(FATAL) << "failed to open perf file for writing";
    }
  }
}

void LogHelper::saveConfig(
    const std::unordered_map<std::string, std::string>& config) {
  if (!isMaster_) {
    return;
  }

  std::ofstream configFile(getRunFile("config", runIdx_, runPath_));
  cereal::JSONOutputArchive ar(configFile);
  ar(CEREAL_NVP(config));
}

void LogHelper::writeHeader(SSLTrainMeters& meters) {
  if (!isMaster_) {
    return;
  }

  std::ofstream perfFile;
  perfFile.open(perfFileName_);
  auto perfMsg = formatStatus(meters, 0, 0, {}, false, true, "\t", true);
  appendToLog(perfFile, "# " + perfMsg);
}

void LogHelper::logStatus(
    SSLTrainMeters& mtrs,
    int64_t epoch,
    int64_t iter,
    const std::unordered_map<std::string, double>& logFields) {
  syncMeter(mtrs);

  if (!isMaster_) {
    return;
  }

  try {
    std::ofstream logFile, perfFile;
    logFile.open(logFileName_, std::ofstream::out | std::ofstream::app);
    perfFile.open(perfFileName_, std::ofstream::out | std::ofstream::app);
    auto logMsg =
        formatStatus(mtrs, epoch, iter, logFields, true, false, " | ", false);
    auto perfMsg =
        formatStatus(mtrs, epoch, iter, logFields, false, true, " ", false);
    LOG_MASTER(INFO) << logMsg;
    appendToLog(logFile, logMsg);
    appendToLog(perfFile, perfMsg);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Error while writing logs: " << ex.what();
  }
}

void LogHelper::saveModel(
    const std::string& tag,
    const std::unordered_map<std::string, std::string>& config,
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
    std::shared_ptr<fl::FirstOrderOptimizer> critoptim,
    std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
    std::shared_ptr<fl::FirstOrderOptimizer> scaleoptim) {
  if (!isMaster_) {
    return;
  }

  try {
    std::string filename =
        getRunFile("model_" + cleanFilepath(tag) + ".bin", runIdx_, runPath_);
    if (noiselm) {
      W2lSerializer::save(
          filename,
          config,
          network,
          criterion,
          netoptim,
          critoptim,
          noiselm->params(),
          scaleoptim);
    } else {
      W2lSerializer::save(
          filename, config, network, criterion, netoptim, critoptim);
    }
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Error while saving models: " << ex.what();
  }
}

void LogHelper::logAndSaveModel(
    SSLTrainMeters& meters,
    const std::unordered_map<std::string, std::string>& config,
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
    std::shared_ptr<fl::FirstOrderOptimizer> critoptim,
    std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
    std::shared_ptr<fl::FirstOrderOptimizer> scaleoptim,
    const std::unordered_map<std::string, double>& logFields) {
  saveModel(
      "last",
      config,
      network,
      criterion,
      netoptim,
      critoptim,
      noiselm,
      scaleoptim);

  int iter = logOnEpoch_ ? std::stoi(config.at(kEpoch))
                         : std::stoi(config.at(kIteration));

  if (FLAGS_itersave) {
    std::string tag =
        logOnEpoch_ ? format("epoch_%04d", iter) : format("iter_%08d", iter);
    if (logOnEpoch_ && iter % FLAGS_saveevery == 0) {
      saveModel(
          tag,
          config,
          network,
          criterion,
          netoptim,
          critoptim,
          noiselm,
          scaleoptim);
    }
  }

  logStatus(
      meters,
      std::stoi(config.at(kEpoch)),
      std::stoi(config.at(kIteration)),
      logFields);

  for (auto& s : meters.valid) {
    double verr = s.second.edits[kTarget].value()[0];
    auto sit = validminerrs_.find(s.first);
    if (sit == validminerrs_.end() || sit->second > verr) {
      validminerrs_[s.first] = verr;
      saveModel(
          s.first,
          config,
          network,
          criterion,
          netoptim,
          critoptim,
          noiselm,
          scaleoptim);
    }
  }
}

std::string LogHelper::formatStatus(
    SSLTrainMeters& meters,
    int64_t epoch,
    int64_t iter,
    const std::unordered_map<std::string, double>& logFields,
    bool verbose /* = false */,
    bool date /* = false */,
    const std::string& separator /* = " " */,
    bool headerOnly /* = false */) {
  std::string header, status;

  auto insertItem = [&](std::string key, std::string val) {
    if (verbose) {
      val = key + ": " + val;
    }
    header = header + (header.empty() ? "" : separator) + key;
    status = status + (status.empty() ? "" : separator) + val;
  };

  auto insertSSLDatasetMeters = [&insertItem](
                                    SSLDatasetMeters& meter, std::string tag) {
    for (auto& m : meter.losses) {
      insertItem(
          tag + "-loss-" + m.first, format("%10.5f", m.second.value()[0]));
    }
    for (auto& m : meter.edits) {
      insertItem(
          tag + "-" + m.first + "ER", format("%5.2f", m.second.value()[0]));
    }
  };
  auto insertNoiselmMeters = [&insertItem](NoiselmMeters& meter) {
    for (auto& m : meter.losses) {
      insertItem(m.first, format("%10.5f", m.second.value()[0]));
    }
  };

  if (date) {
    insertItem("date", format("%s", getCurrentDate().c_str()));
    insertItem("time", format("%s", getCurrentTime().c_str()));
  }

  insertItem("epoch", format("%8d", epoch));
  insertItem("iter", format("%8d", iter));

  insertItem(
      "lr-net", headerOnly ? "" : format("%4.6lf", logFields.at("lr-net")));
  insertItem(
      "lr-crit", headerOnly ? "" : format("%4.6lf", logFields.at("lr-crit")));
  insertItem(
      "lr-sc", headerOnly ? "" : format("%4.6lf", logFields.at("lr-sc")));
  insertItem(
      "sc-noise", headerOnly ? "" : format("%4.6lf", logFields.at("sc-noise")));

  int rt = meters.timer[kRuntime].value();
  insertItem(
      kRuntime,
      format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60));

  for (auto& m : meters.timer) {
    if (m.first == kRuntime) {
      continue;
    }
    insertItem(m.first + "(ms)", format("%.2f", m.second.value() * 1000));
  }

  insertSSLDatasetMeters(meters.train, "train");
  for (auto& v : meters.valid) {
    insertSSLDatasetMeters(v.second, v.first);
  }
  insertNoiselmMeters(meters.noiselm);

  auto stats_unp = meters.statsUnpaired.value();
  auto numsamples_unp = std::max<int64_t>(stats_unp[4], 1);
  auto isztotal_unp = stats_unp[0];

  auto stats_noise = meters.statsNoise.value();
  auto numsamples_noise = std::max<int64_t>(stats_noise[4], 1);
  auto isztotal_noise = stats_noise[0];

  auto stats = meters.stats.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem(
      "avg-isz",
      format(
          "%03d", (isztotal + isztotal_unp) / (numsamples + numsamples_unp)));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec = audioProcSec * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }

  double audioProcSec_unp = isztotal_unp * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec_unp = audioProcSec_unp * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec_unp /= FLAGS_samplerate;
  }

  double audioProcSec_noise = isztotal_noise * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec_noise = audioProcSec_noise * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec_noise /= FLAGS_samplerate;
  }

  auto worldSize = fl::getWorldSize();
  double timeTakenSec =
      meters.timer[kTimer].value() * (numsamples + numsamples_unp) / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem("hrs_unp", format("%7.2f", audioProcSec_unp / 3600.0));
  insertItem("hrs_noise", format("%7.2f", audioProcSec_noise / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0
          ? format("%.2f", (audioProcSec + audioProcSec_unp) / timeTakenSec)
          : "n/a");

  return headerOnly ? header : status;
}

template <>
void syncMeter<SSLTrainMeters>(SSLTrainMeters& meters) {
  syncMeter(meters.stats);
  syncMeter(meters.statsNoise);
  syncMeter(meters.statsUnpaired);
  for (auto& m : meters.timer) {
    syncMeter(m.second);
  }
  syncMeter(meters.train);
  for (auto& m : meters.valid) {
    syncMeter(m.second);
  }
  syncMeter(meters.noiselm);
}

template <>
void syncMeter<SSLDatasetMeters>(SSLDatasetMeters& meters) {
  for (auto& m : meters.edits) {
    syncMeter(m.second);
  }
  for (auto& m : meters.losses) {
    syncMeter(m.second);
  }
}

template <>
void syncMeter<NoiselmMeters>(NoiselmMeters& meters) {
  for (auto& m : meters.losses) {
    syncMeter(m.second);
  }
}

void resetTimeStatMeters(SSLTrainMeters& meters) {
  for (auto& m : meters.timer) {
    m.second.reset();
  }
  meters.stats.reset();
  meters.statsNoise.reset();
  meters.statsUnpaired.reset();
}

void stopTimeMeters(SSLTrainMeters& meters) {
  for (auto& m : meters.timer) {
    m.second.stop();
  }
}

void resetDatasetMeters(SSLDatasetMeters& meters) {
  for (auto& m : meters.edits) {
    m.second.reset();
  }
  for (auto& m : meters.losses) {
    m.second.reset();
  }
}

void resetNoiselmMeters(NoiselmMeters& meters) {
  for (auto& m : meters.losses) {
    m.second.reset();
  }
}

// For Train.cpp
template <>
void syncMeter<NoiseTrainMeters>(NoiseTrainMeters& mtrs) {
  syncMeter(mtrs.stats);
  syncMeter(mtrs.runtime);
  syncMeter(mtrs.timer);
  syncMeter(mtrs.fwdtimer);
  syncMeter(mtrs.critfwdtimer);
  syncMeter(mtrs.bwdtimer);
  syncMeter(mtrs.optimtimer);
  syncMeter(mtrs.train.tknEdit);
  syncMeter(mtrs.train.wrdEdit);
  syncMeter(mtrs.train.loss);
  syncMeter(mtrs.train.wLER);
  for (auto& v : mtrs.valid) {
    syncMeter(v.second.tknEdit);
    syncMeter(v.second.wrdEdit);
    syncMeter(v.second.loss);
  }
}

} // namespace w2l
