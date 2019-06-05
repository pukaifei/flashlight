/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include <string>
#include <vector>

namespace w2l {

class LMCritic : public fl::Container {
 public:
  LMCritic(
      std::shared_ptr<fl::Module> network,
      const std::vector<int>& dictIndexMap,
      int numDictPadding,
      int startIndex,
      int unkIndex = -1,
      bool useGumbel = false,
      double gumbelTemperature = 0.0);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;

  double getTemperature();

  void setTemperature(double t);

 private:
  af::array dictIndexMap_;
  fl::Variable startProb_;
  int numDictPadding_, unkIndex_;
  bool useGumbel_;
  double gumbelTemperature_;

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      dictIndexMap_,
      startProb_,
      numDictPadding_,
      unkIndex_,
      useGumbel_,
      gumbelTemperature_)

  LMCritic() = default;

  fl::Variable preprocessInput(fl::Variable input);

  std::shared_ptr<fl::Module> lmNetwork() const {
    return module(0);
  }
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::LMCritic)
