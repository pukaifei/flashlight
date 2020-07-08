/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "criterion/criterion.h"
#include "module/module.h"
#include "runtime/runtime.h"

DEFINE_string(model_path, "", "Path to the model to convert");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  std::shared_ptr<fl::Module> network;
  std::shared_ptr<w2l::SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;

  gflags::ParseCommandLineFlags(&argc, &argv, false);

  if (FLAGS_model_path.empty()) {
    throw std::invalid_argument("Model path cannot be empty");
  }

  // Read gflags from old model
  LOG(INFO) << "Loading model from " << FLAGS_model_path;
  w2l::W2lSerializer::load(FLAGS_model_path, cfg, network, criterion);
  auto flags = cfg.find(w2l::kGflags);
  LOG_IF(FATAL, flags == cfg.end()) << "Invalid config loaded";
  LOG(INFO) << "Reading flags...";
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  LOG(INFO) << "Modifying deprecated flags...";
  // Implicitly sets new flags equal to corresponding deprecated flags
  w2l::handleDeprecatedFlags();

  // Save flags. Won't serialize deprecated flags
  cfg[w2l::kGflags] = w2l::serializeGflags();

  // Save model
  auto newModelPath = FLAGS_model_path + ".new";
  LOG(INFO) << "Saving model to " << newModelPath;
  w2l::W2lSerializer::save(newModelPath, cfg, network, criterion);

  LOG(INFO) << "Done.";
  return 0;
}
