/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>

#include <flashlight/flashlight.h>

#include "criterion/criterion.h"
#include "experimental/semisupervised/src/module/LMCritic.h"
#include "experimental/semisupervised/src/runtime/Defines.h"
#include "runtime/Logger.h"

namespace w2l {
struct SSLDatasetMeters {
  std::map<std::string, fl::EditDistanceMeter> edits;
  std::map<std::string, fl::AverageValueMeter> losses;

  SSLDatasetMeters()
      : edits({{kTarget, fl::EditDistanceMeter()},
               {kWord, fl::EditDistanceMeter()}}),
        losses({{kASR, fl::AverageValueMeter()},
                {kLM, fl::AverageValueMeter()},
                {kFullModel, fl::AverageValueMeter()}}) {}
};

struct SSLTrainMeters {
  std::map<std::string, fl::TimeMeter> timer;
  SSLDatasetMeters train;
  std::map<std::string, SSLDatasetMeters> valid;
  SpeechStatMeter stats;

  SSLTrainMeters()
      : timer({{kRuntime, fl::TimeMeter(false)},
               {kTimer, fl::TimeMeter(true)},
               {kSampleTimer, fl::TimeMeter(true)},
               {kFwdTimer, fl::TimeMeter(true)},
               {kCritFwdTimer, fl::TimeMeter(true)},
               {kLMCritFwdTimer, fl::TimeMeter(true)},
               {kBwdTimer, fl::TimeMeter(true)},
               {kOptimTimer, fl::TimeMeter(true)}}) {}
};

class LogHelper {
 public:
  LogHelper(int runIdx, std::string runPath, bool isMaster, bool logOnEpoch);

  void saveConfig(const std::unordered_map<std::string, std::string>& config);

  void writeHeader(SSLTrainMeters& meters);

  void logStatus(
      SSLTrainMeters& mtrs,
      int64_t epoch,
      const std::unordered_map<std::string, double>& logFields);

  void saveModel(
      const std::string& tag,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<LMCritic> lmcrit,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim);

  void logAndSaveModel(
      SSLTrainMeters& meters,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<LMCritic> lmcrit,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
      const std::unordered_map<std::string, double>& logFields);

  std::string formatStatus(
      SSLTrainMeters& meters,
      int64_t epoch,
      const std::unordered_map<std::string, double>& logFields,
      bool verbose = false,
      bool date = false,
      const std::string& separator = " ",
      bool headerOnly = false);

 private:
  int runIdx_;
  std::string runPath_;
  bool isMaster_, logOnEpoch_;
  std::string logFileName_, perfFileName_;
  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs_;

  LogHelper() {}
};

template <>
void syncMeter<SSLTrainMeters>(SSLTrainMeters& meters);

template <>
void syncMeter<SSLDatasetMeters>(SSLDatasetMeters& meters);

void resetTimeStatMeters(SSLTrainMeters& meters);

void stopTimeMeters(SSLTrainMeters& meters);

void resetDatasetMeters(SSLDatasetMeters& meters);
} // namespace w2l
