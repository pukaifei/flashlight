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
#include "experimental/semisupervised/runtime/Defines.h"
#include "runtime/Logger.h"

namespace w2l {
struct SSLDatasetMeters {
  std::map<std::string, fl::EditDistanceMeter> edits;
  std::map<std::string, fl::AverageValueMeter> losses;

  SSLDatasetMeters()
      : edits({{kTarget, fl::EditDistanceMeter()}}),
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
               {kBwdTimer, fl::TimeMeter(true)},
               {kOptimTimer, fl::TimeMeter(true)}}) {}
};

class LogHelper {
 public:
  LogHelper(int runIdx, std::string runPath, bool isMaster, bool logOnEpoch);

  void saveConfig(const std::unordered_map<std::string, std::string>& config);

  void writeHeader(SSLTrainMeters& meters);

  void logStatus(SSLTrainMeters& mtrs, int64_t epoch, double lr);

  void saveModel(
      const std::string& tag,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim);

  void logAndSaveModel(
      SSLTrainMeters& meters,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim);

  std::pair<std::string, std::string> formatStatus(
      SSLTrainMeters& meters,
      int64_t epoch,
      double lr,
      bool verbose = false,
      bool date = false,
      const std::string& separator = " ");

 private:
  int runIdx_;
  std::string runPath_;
  bool isMaster_, logOnEpoch_;
  std::ofstream logFile_, perfFile_;
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
