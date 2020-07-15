/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/lead2Gold/src/runtime/Eval.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "experimental/lead2Gold/src/common/Utils.h"
#include "experimental/lead2Gold/src/data/Utils.h"
#include "experimental/lead2Gold/src/runtime/Logger.h"

namespace w2l {

void evalOutput(
    const af::array& op,
    const af::array& target,
    std::map<std::string, fl::EditDistanceMeter>& mtr,
    const Dictionary& tgtDict,
    std::shared_ptr<SequenceCriterion> criterion) {
  auto batchsz = op.dims(2);
  for (int b = 0; b < batchsz; ++b) {
    auto tgt = target(af::span, b);
    auto viterbipath =
        afToVector<int>(criterion->viterbiPath(op(af::span, af::span, b)));
    auto tgtraw = afToVector<int>(tgt);

    // uniq(viterbipath);

    // Remove `-1`s appended to the target for batching (if any)
    auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
    tgtraw.resize(labellen);

    // remapLabels(viterbipath, tgtDict);
    // remapLabels(tgtraw, tgtDict);

    auto ltrPred = NoisetknPrediction2Ltr(viterbipath, tgtDict);
    auto ltrTgt = tknTarget2Ltr(tgtraw, tgtDict);
    auto wrdPred = tkn2Wrd(ltrPred);
    auto wrdTgt = tkn2Wrd(ltrTgt);

    mtr[kTarget].add(ltrPred, ltrTgt);
    mtr[kWord].add(wrdPred, wrdTgt);
  }
}

void evalDataset(
    std::shared_ptr<fl::Module> ntwrk,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<AutoSegBeamNoiseCriterion> asgbeamnoisecrit,
    std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
    int replabel,
    std::shared_ptr<W2lDataset> testds,
    SSLDatasetMeters& mtrs,
    DictionaryMap& dicts,
    bool evalbeamnoise) {
  resetDatasetMeters(mtrs);
  for (auto& sample : *testds) {
    auto output = ntwrk->forward({fl::input(sample[kInputIdx])}).front();
    // Compute paired metrics
    auto lossPaired = criterion->forward(
        {output, fl::Variable(sample[kTargetIdx], false)})[0];
    mtrs.losses[kASRPaired].add(lossPaired.array());

    if (noiselm && evalbeamnoise) {
      std::vector<af::array> updatedTranscripts;
      updatedTranscripts = getUpdateTrancripts(output, criterion, dicts);
      auto nullVar = fl::Variable();
      auto lossUnpaired = asgbeamnoisecrit
                              ->forward(
                                  output,
                                  nullVar,
                                  criterion->param(0),
                                  fl::noGrad(updatedTranscripts[0]),
                                  fl::noGrad(updatedTranscripts[1]))
                              .front();
      mtrs.losses[kASRUnpaired].add(lossUnpaired.array());
    }

    evalOutput(
        output.array(),
        sample[kTargetIdx],
        mtrs.edits,
        dicts[kTargetIdx],
        criterion);
  }
}

void runEval(
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<AutoSegBeamNoiseCriterion> asgbeamnoisecrit,
    std::shared_ptr<NoiseLMLetterSwapUnit> noiselm,
    int replabel,
    const std::unordered_map<std::string, std::shared_ptr<W2lDataset>>& ds,
    SSLTrainMeters& meters,
    DictionaryMap& dicts,
    bool evalbeamnoise) {
  network->eval();
  criterion->eval();
  for (auto& d : ds) {
    evalDataset(
        network,
        criterion,
        asgbeamnoisecrit,
        noiselm,
        replabel,
        d.second,
        meters.valid[d.first],
        dicts,
        evalbeamnoise);
  }
}

} // namespace w2l
