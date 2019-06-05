/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/semisupervised/src/runtime/DataScheduler.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "experimental/semisupervised/src/runtime/Defines.h"
#include "runtime/Logger.h"

namespace w2l {

DataScheduler::DataScheduler(
    const std::vector<std::shared_ptr<W2lDataset>>& datasets,
    const std::vector<int64_t>& dataTypes,
    const std::vector<int64_t>& numIters,
    int64_t curEpoch /* = 1 */)
    : ds_(datasets.begin(), datasets.end()),
      dataTypes_(dataTypes.begin(), dataTypes.end()),
      dsNumIters_(numIters.begin(), numIters.end()),
      dsCurIter_(ds_.size(), 0),
      dsIterOffset_(ds_.size(), 0),
      dsCurEpochs_(ds_.size(), curEpoch) {
  initialize();

  if (!FLAGS_noresample) {
    for (auto& d : ds_) {
      LOG_MASTER(INFO) << "Shuffling trainset";
      d->shuffle(curEpoch);
    }
  }
}

void DataScheduler::initialize() {
  LOG_IF(
      FATAL,
      ds_.size() != dsNumIters_.size() || ds_.size() != dataTypes_.size())
      << "mismatch between the number of datasets "
      << "and the number of schedules or data types specified";

  curDs_ = 0;
  while (curDs_ < dsNumIters_.size() && dsNumIters_[curDs_] == 0) {
    ++curDs_;
  }
  LOG_IF(FATAL, curDs_ == dsNumIters_.size())
      << "Invalid training schedule (zero iterations on all datasets)";
}

std::vector<af::array> DataScheduler::get() {
  auto idx = (dsIterOffset_[curDs_] + dsCurIter_[curDs_]) % ds_[curDs_]->size();
  auto sample = ds_[curDs_]->get(idx);
  auto globalBatchIdx = ds_[curDs_]->getGlobalBatchIdx(idx);
  sample.emplace_back(af::constant(dataTypes_[curDs_], 1, s64));
  sample.emplace_back(af::constant(globalBatchIdx, 1, s64));

  update();
  return sample;
}

void DataScheduler::update() {
  ++dsCurIter_[curDs_];

  if (!FLAGS_noresample &&
      (dsIterOffset_[curDs_] + dsCurIter_[curDs_]) % ds_[curDs_]->size() == 0) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    ds_[curDs_]->shuffle(++dsCurEpochs_[curDs_] /* seed */);
  }

  // switch the dataset for the next iteration if necessary
  if (dsCurIter_[curDs_] % dsNumIters_[curDs_] == 0) {
    curDs_ = (curDs_ + 1) % ds_.size();
    while (dsNumIters_[curDs_] == 0) {
      curDs_ = (curDs_ + 1) % ds_.size();
    }
  }
}

std::vector<int64_t> DataScheduler::getSchedule() {
  return dsNumIters_;
}

void DataScheduler::setSchedule(std::vector<int64_t> newIters) {
  dsNumIters_ = std::move(newIters);
  initialize();
  for (int i = 0; i < dsCurIter_.size(); ++i) {
    dsIterOffset_[i] = (dsIterOffset_[i] + dsCurIter_[i]) % ds_[i]->size();
    dsCurIter_[i] = 0;
  }
}
} // namespace w2l
