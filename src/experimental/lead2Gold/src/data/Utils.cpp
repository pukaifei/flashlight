/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "experimental/lead2Gold/src/data/Utils.h"
#include "experimental/lead2Gold/src/common/Utils.h"


namespace w2l {

std::vector<af::array> getUpdateTrancripts(fl::Variable& emissions, std::shared_ptr<SequenceCriterion> criterion, DictionaryMap& dicts, int padidx, bool addeos){
  auto viterbipaths_af = criterion->viterbiPath(emissions.array()); 
  //auto viterbipaths_af = w2l::viterbiPath(emissions.array(), transitions.array()); 
  auto viterbipaths = afToVector<int>(viterbipaths_af);
  int maxsizeViterbi = viterbipaths_af.dims(0);
  int batchsz = viterbipaths_af.dims(1);

  std::vector<std::vector<int>> new_transcriptions(batchsz, std::vector<int> {});
   
  int maxsize = 0;
  for (int b = 0; b < batchsz; b++) {
    std::vector<int> viterbipath(viterbipaths.begin() + b*maxsizeViterbi, viterbipaths.begin() + (b+1)*maxsizeViterbi); 
    uniq(viterbipath);
    if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kCtcBeamNoiseCriterion) {
      int blankIdx = dicts[kTargetIdx].getIndex(kBlankToken);
      viterbipath.erase(
          std::remove(viterbipath.begin(), viterbipath.end(), blankIdx), viterbipath.end());
    }
    remapLabels(viterbipath, dicts[kTargetIdx]);
    if (addeos){
      viterbipath.push_back(dicts[kNoisyNoiselmKeyIdx].getIndex(kEosToken));
    }
    maxsize = std::max(maxsize, (int)viterbipath.size());
    new_transcriptions[b] = viterbipath;
  }

  auto new_transcriptions_flat = std::vector<int>(maxsize*batchsz, padidx);
  for (int b = 0; b < batchsz; b++) {
    std::copy(new_transcriptions[b].begin(), new_transcriptions[b].begin() + new_transcriptions[b].size(), new_transcriptions_flat.begin() + b*maxsize);
  }

  auto new_transcriptions_flat_unpack = unpackReplabels(new_transcriptions_flat, dicts[kTargetIdx], FLAGS_replabel);

  auto new_transcriptions_af = af::array(maxsize, batchsz, new_transcriptions_flat.data());
  auto new_transcriptions_af_unpack = af::array(maxsize, batchsz, new_transcriptions_flat_unpack.data());

  return {new_transcriptions_af, new_transcriptions_af_unpack};
}

//does not work. do not use. Use ds->updateTargets(idx, newTranscriptions).
void updateTrancripts(std::vector<af::array>& sample, fl::Variable& emissions, std::shared_ptr<SequenceCriterion> criterion, DictionaryMap& dicts){
  auto updatedTranscripts = getUpdateTrancripts(emissions, criterion, dicts);
  sample[kTargetIdx] = updatedTranscripts[0];
  sample[kNoiseKeyIdx] = updatedTranscripts[1];
}

//return a batch of words
std::vector<std::vector<std::string>> getUpdateTrancriptsWords(fl::Variable& emissions, std::shared_ptr<SequenceCriterion> criterion, DictionaryMap& dicts){
  auto viterbipaths_af = criterion->viterbiPath(emissions.array()); 
  //auto viterbipaths_af = w2l::viterbiPath(emissions.array(), transitions.array()); 
  auto viterbipaths = afToVector<int>(viterbipaths_af);
  int maxsizeViterbi = viterbipaths_af.dims(0); // T
  int batchsz = viterbipaths_af.dims(1);

  std::vector<std::vector<std::string>> new_transcriptions(batchsz);
   
  for (int b = 0; b < batchsz; b++) {
    std::vector<int> viterbipath(viterbipaths.begin() + b*maxsizeViterbi, viterbipaths.begin() + (b+1)*maxsizeViterbi); 
    auto ltrPred = NoisetknPrediction2Ltr(viterbipath, dicts[kTargetIdx]);
    new_transcriptions[b] = tkn2Wrd(ltrPred);
  }

  return new_transcriptions;
}



} // namespace w2l
