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



#include "runtime/SpeechStatMeter.h"


#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "runtime/Logger.h"
#include "experimental/lead2Gold/src/common/Defines.h"


#define LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

namespace w2l {

//For Train.cpp
struct NoiseDatasetMeters {
  fl::EditDistanceMeter tknEdit, wrdEdit;
  fl::AverageValueMeter loss;
  fl::AverageValueMeter wLER;
};

//For Train.cpp
struct NoiseTrainMeters {
  fl::TimeMeter runtime;
  fl::TimeMeter timer{true};
  fl::TimeMeter sampletimer{true};
  fl::TimeMeter fwdtimer{true}; // includes network + criterion time
  fl::TimeMeter critfwdtimer{true};
  fl::TimeMeter bwdtimer{true}; // includes network + criterion time
  fl::TimeMeter optimtimer{true};

  NoiseDatasetMeters train;
  std::map<std::string, NoiseDatasetMeters> valid;

  SpeechStatMeter stats;
};

struct SSLDatasetMeters {
  std::map<std::string, fl::EditDistanceMeter> edits;
  std::map<std::string, fl::AverageValueMeter> losses;

  SSLDatasetMeters()
      : edits({ {kTarget, fl::EditDistanceMeter()},
                {kWord, fl::EditDistanceMeter()}
              }),
        losses({ {kASRPaired, fl::AverageValueMeter()},
                 {kASRUnpaired, fl::AverageValueMeter()}     
              }) {}             
};

struct NoiselmMeters {
  std::map<std::string, fl::AverageValueMeter> losses;

  NoiselmMeters()
      : losses({ {klossScale, fl::AverageValueMeter()},
                 {klossNoiselm, fl::AverageValueMeter()}
                      }) {}           
};

struct SSLTrainMeters {
  std::map<std::string, fl::TimeMeter> timer;
  SSLDatasetMeters train;
  std::map<std::string, SSLDatasetMeters> valid;
  NoiselmMeters noiselm;

  SpeechStatMeter stats;
  SpeechStatMeter statsNoise;
  SpeechStatMeter statsUnpaired;

  SSLTrainMeters()
      : timer({{kRuntime, fl::TimeMeter(false)},
               {kTimer, fl::TimeMeter(true)},
               {kSampleTimer, fl::TimeMeter(true)},
               {kFwdTimer, fl::TimeMeter(true)},
               {kCritFwdTimer, fl::TimeMeter(true)},
               {kCritFwdNoiseTimer, fl::TimeMeter(true)},
               {kUpdateTransTimer, fl::TimeMeter(true)},
               {kUpdateScaleTimer, fl::TimeMeter(true)},
               {kUpdateNoiseModelTimer, fl::TimeMeter(true)},
               {kBwdTimer, fl::TimeMeter(true)},
               {kOptimTimer, fl::TimeMeter(true)}}) {}
};

//For Train.cpp
std::pair<std::string, std::string> getStatus(
    NoiseTrainMeters& meters,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    bool verbose = false,
    bool date = false,
    const std::string& separator = " ");

class LogHelper {
 public:
  LogHelper(int runIdx, std::string runPath, bool isMaster, bool logOnEpoch);

  void saveConfig(const std::unordered_map<std::string, std::string>& config);

  void writeHeader(SSLTrainMeters& meters);

  void logStatus(
      SSLTrainMeters& mtrs,
      int64_t epoch,
      int64_t iter,
      const std::unordered_map<std::string, double>& logFields);

  void saveModel(
      const std::string& tag,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
      std::shared_ptr<fl::FirstOrderOptimizer> critoptim,
      std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
      std::shared_ptr<fl::FirstOrderOptimizer> scaleoptim);

  void logAndSaveModel(
      SSLTrainMeters& meters,
      const std::unordered_map<std::string, std::string>& config,
      std::shared_ptr<fl::Module> network,
      std::shared_ptr<SequenceCriterion> criterion,
      std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
      std::shared_ptr<fl::FirstOrderOptimizer> critoptim,
      std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
      std::shared_ptr<fl::FirstOrderOptimizer> scaleoptim,
      const std::unordered_map<std::string, double>& logFields);

  std::string formatStatus(
      SSLTrainMeters& meters,
      int64_t epoch,
      int64_t iter,
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

template <>
void syncMeter<NoiselmMeters>(NoiselmMeters& meters);


void resetTimeStatMeters(SSLTrainMeters& meters);

void stopTimeMeters(SSLTrainMeters& meters);

void resetDatasetMeters(SSLDatasetMeters& meters);

void resetNoiselmMeters(NoiselmMeters& meters);

//for Train.cpp
template <>
void syncMeter<NoiseTrainMeters>(NoiseTrainMeters& mtrs);

} // namespace w2l
