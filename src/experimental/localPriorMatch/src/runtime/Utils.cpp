/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <arrayfire.h>
#include <glog/logging.h>

#include "experimental/localPriorMatch/src/runtime/Utils.h"

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "experimental/localPriorMatch/src/runtime/Defines.h"

namespace w2l {

Dictionary createFairseqTokenDict(const std::string& filepath) {
  Dictionary dict;

  dict.addEntry("<fairseq_style>", 0);
  dict.addEntry("<pad>", 1);
  dict.addEntry(kEosToken, 2);
  dict.addEntry(kUnkToken, 3);

  if (filepath.empty()) {
    throw std::runtime_error("Empty filepath specified for token dictiinary.");
    return dict;
  }
  std::ifstream infile(trim(filepath));
  if (!infile) {
    throw std::runtime_error("Unable to open dictionary file: " + filepath);
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    if (!tkns.empty()) {
      dict.addEntry(tkns[0]);
    }
  }

  return dict;
}

std::pair<std::vector<int>, int> genTokenDictIndexMap(
    const Dictionary& dict1,
    const Dictionary& dict2) {
  int size1 = dict1.indexSize();
  int size2 = dict2.indexSize();

  std::vector<int> mapping(size1);
  int numPadding = 0;

  for (int idx1 = 0; idx1 < size1; ++idx1) {
    auto token = dict1.getEntry(idx1);
    auto idx2 = dict2.getIndex(token);
    if (idx2 < size2) {
      mapping[idx1] = idx2;
    } else { // assume we already ran
             // `dict2.setDefaultIndex(dict2.indexSize());`
      mapping[idx1] = size2 + numPadding;
      ++numPadding;
    }
  }

  return std::make_pair(mapping, numPadding);
}

void print_path(std::vector<int> path, Dictionary& dict) {
  std::cout << "Idx = "
            << stringify<int>(path)
            << std::endl;
  std::cout << "Str = " 
            << stringify<std::string>(wrdIdx2Wrd(path, dict)) 
            << std::endl;
}

std::vector<int> remapLabelsForLM(std::vector<int> path, Dictionary& dict) {
  // remove all trailing <eos> and then add one back
  remapLabels(path, dict);
  path.push_back(dict.getIndex(kEosToken));
  return path;
}

std::string arrDimStr(const af::array& arr) {
  std::ostringstream os;
  auto dims = arr.dims();
  os << "(" 
     << dims[0] << ", " << dims[1] << ", " 
     << dims[2] << ", " << dims[3] << ")";
  return os.str();
}

af::array getTargetLength(af::array& target, int eosIdx) {
  return af::sum(target != eosIdx, 0).as(af::dtype::s32) + 1;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> batchBeamSearch(
    const fl::Variable& output, 
    const std::shared_ptr<Seq2SeqCriterion>& criterion) {
  criterion->eval();
  std::vector<std::vector<int>> paths;
  std::vector<float> scores;
  std::vector<int> hypoNums;
  for (int b = 0; b < output.dims(2); b++) {
    std::vector<Seq2SeqCriterion::CandidateHypo> initBeam;
    initBeam.emplace_back(Seq2SeqCriterion::CandidateHypo{});
    // TODO: WARNING: this is a transition version that uses very slow beam
    // search call from Seq2SeqCriterion
    auto hypos = criterion->beamSearch(
        output.array()(af::span, af::span, b),
        initBeam, FLAGS_pmBeamsz, FLAGS_maxdecoderoutputlen);

    for (auto& hypo : hypos) {
      paths.push_back(hypo.path);
      scores.push_back(hypo.score);
    }
    hypoNums.push_back(hypos.size());
  }

  if (FLAGS_debug) {
    std::ostringstream os;
    os << "lengths : ";
    for (auto& path : paths) { os << path.size() << " "; }
    os << "\nscores : ";
    for (auto& score : scores) { os << score << " "; }
    LOG(INFO) << std::endl << os.str();
  }

  criterion->train();
  return std::make_pair(paths, hypoNums);
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> filterBeamByLength(
    const std::vector<std::vector<int>>& paths, 
    const std::vector<int>& hypoNums,
    const std::vector<int>& refLengths) {
  if (hypoNums.size() != refLengths.size()) {
    LOG(FATAL) << "size of hypoNums(" << hypoNums.size() << ") and refLengths("
               << refLengths.size() << ") does not match";
  }
  float rlb = (FLAGS_hyplenratiolb < 0) ? 0 : FLAGS_hyplenratiolb;
  float rub = (FLAGS_hyplenratioub < 0) ? 999 : FLAGS_hyplenratioub;
  if (FLAGS_debug) {
    LOG(INFO) << "refLen : " << stringify<int>(refLengths);
  }

  int offset = 0;
  std::vector<std::vector<int>> newPaths;
  std::vector<int> newHypoNums;
  for (int b=0; b<hypoNums.size(); b++) {
    int newHypoNum = 0;
    int lb = std::floor(rlb * refLengths[b]);
    int ub = std::ceil(rub * refLengths[b]);
    for (int i=0; i<hypoNums[b]; i++) {
      int cur_idx = offset + i;
      // also remove length=1 (empty hypotheses)
      if (paths[cur_idx].size() >= lb 
          && paths[cur_idx].size() <= ub
          && paths[cur_idx].size() > 1) {
        newPaths.push_back(paths[cur_idx]);
        newHypoNum += 1;
      }
    }
    offset += hypoNums[b];
    newHypoNums.push_back(newHypoNum);
  }

  return std::make_pair(newPaths, newHypoNums);
}

fl::Variable computeLmLogprob(
    const std::vector<std::vector<int>>& paths,
    const std::shared_ptr<LMCritic>& lmcrit,
    const Dictionary& dict) {

  auto tgt = batchTarget(paths, dict.getIndex(kEosToken));
  auto tgtOnehot = fl::noGrad(makeOnehot(tgt, dict.indexSize()));
  auto tgtLen = af::array(
      af::dim4(1, paths.size()), getLengths<int, int>(paths).data());
  auto lmLogprob = fl::negate(
      lmcrit->forward({fl::log(tgtOnehot), fl::noGrad(tgtLen)}).front());
  return lmLogprob;
}

fl::Variable postprocLmLogprob(
    fl::Variable logprob, const std::vector<std::vector<int>>& paths) {
  fl::Variable procLogprob;
  if (FLAGS_normlmcritprob == kNoNorm) {
    procLogprob = logprob;
  } else if (FLAGS_normlmcritprob == kLenNorm) {
    std::vector<float> lengthsVec;
    for (auto& path : paths) {
      lengthsVec.emplace_back(path.size());
    }
    auto lengths = af::array(logprob.dims(), lengthsVec.data());
    auto divisor = af::root(FLAGS_lmcritsmooth, lengths);
    if (FLAGS_debug) {
      af::print("divisor", af::transpose(divisor));
    }
    procLogprob = logprob / fl::noGrad(divisor);
  } else {
    LOG(FATAL) << "Unsupported normalizing method : " << FLAGS_normlmcritprob;
  }
  return procLogprob;
}

fl::Variable computeS2SLogprob(
    const std::vector<std::vector<int>>& paths,
    const std::vector<int>& hypoNums,
    const fl::Variable& encoderOutput,
    const std::shared_ptr<Seq2SeqCriterion>& criterion,
    const Dictionary& dict) {
  // batch targets
  auto tgts = fl::noGrad(batchTarget(paths, dict.getIndex(kEosToken)));
  // af::print("Batched Target", tgts.array());
  
  // tile and batch encoder outputs
  std::vector<fl::Variable> tiledEncoderOutputVec;
  for (int i = 0; i < hypoNums.size(); i++) {
    if (hypoNums[i] > 0) {
      auto curOutpt = encoderOutput(af::span, af::span, i);
      auto curTiledOutput = fl::tile(curOutpt, af::dim4(1, 1, hypoNums[i]));
      if (FLAGS_debug) {
        std::cout << " hypoNum[" << i << "] : " << hypoNums[i]
                  << " tiledEncoderOutputVec[" << i << "] : " 
                  << arrDimStr(curTiledOutput.array()) << std::endl;
      }
      tiledEncoderOutputVec.emplace_back(curTiledOutput);
    }
  }
  auto tiledEncoderOutput = concatenate(tiledEncoderOutputVec, 2);
  if (FLAGS_debug) {
      std::cout << " tiledEncoderOutput : " 
                << arrDimStr(tiledEncoderOutput.array()) << std::endl;

  }
  if (paths.size() != static_cast<size_t>(tiledEncoderOutput.dims()[2])) {
      LOG(FATAL) << "Shape mismatch : " << paths.size() 
                 << " vs " << tiledEncoderOutput.dims()[2];
  }
  
  // [sum(hypoNums), 1]
  fl::Variable s2sLoss;
  if (!FLAGS_pmLabelSmooth) {
    // temporarily disable label smoothing
    criterion->setLabelSmooth(0.0);
    s2sLoss = criterion->forward({tiledEncoderOutput, tgts}).front();
    criterion->setLabelSmooth(FLAGS_labelsmooth);
  } else {
    s2sLoss = criterion->forward({tiledEncoderOutput, tgts}).front();
  }
  return (-1 * s2sLoss);
}

fl::Variable postprocS2SLogprob(
    fl::Variable logprob, const std::vector<std::vector<int>>& paths, 
    const std::vector<int>& hypoNums) {
  fl::Variable procLogprob;
  if (FLAGS_norms2sprob == kNoNorm) {
    procLogprob = logprob;
  } else if (FLAGS_norms2sprob == kLenNorm) {
    double maxlen = 1;
    for (auto& path : paths) {
      maxlen = std::max(maxlen, static_cast<double>(path.size()));
    }
    procLogprob = logprob / maxlen;
  } else if (FLAGS_norms2sprob == kUnitNorm) {
    logprob = logprob - fl::noGrad(variableMax(logprob, hypoNums, true).array());
    auto logsumexp = fl::log(variableSum(fl::exp(logprob), hypoNums, true));
    procLogprob = logprob - fl::noGrad(logsumexp.array());
  } else if (FLAGS_norms2sprob != kNoNorm) {
    LOG(FATAL) << "Unsupported norms2sprob : " << FLAGS_norms2sprob;
  }
  return procLogprob;
}

// random re-assign LM probability for ablation study
fl::Variable shuffleProb(
    const fl::Variable& logprob, const std::vector<int>& hypoNums) {
  std::vector<fl::Variable> outputVec;
  int offset = 0;
  for (auto& hypoNum : hypoNums) {
    if (hypoNum > 0) {
      auto logprobSlice = logprob(af::seq(offset, offset+hypoNum-1));
      std::vector<int> idx(hypoNum);
      std::iota(idx.begin(), idx.end(), 0);
      std::random_shuffle(idx.begin(), idx.end());
      af::array idxArr(af::dim4(hypoNum), idx.data());
      outputVec.emplace_back(logprobSlice(idxArr));
      if (FLAGS_debug) {
        std::cout << "Shuffle Index : " << stringify<int>(idx) << std::endl;
      }
      offset += hypoNum;
    }
  }
  if (offset != logprob.dims()[0]) {
    LOG(FATAL) << "Total number of hypos inconsistent : "
               << offset << " vs " << logprob.dims()[0];
  }
  return concatenate(outputVec, 0);

}

// maybe renormalize and/or change to linear scale
fl::Variable adjustProb(
    const fl::Variable& logprob, const std::vector<int>& hypoNums,
    bool renormalize, bool linear) {
  if (!renormalize && !linear) {
    return logprob;
  }

  std::vector<fl::Variable> outputVec;
  int offset = 0;
  for (auto& hypoNum : hypoNums) {
    if (hypoNum > 0) {
      auto logprobSlice = logprob(af::seq(offset, offset+hypoNum-1));
      if (renormalize && linear) {
        outputVec.emplace_back(fl::softmax(logprobSlice, 0));
      } else if (renormalize && !linear) {
        outputVec.emplace_back(fl::logSoftmax(logprobSlice, 0));
      } else if (!renormalize && linear) {
        outputVec.emplace_back(fl::exp(logprobSlice));
      } else {
        LOG(FATAL) << "Something is really wrong. Should never arrive here";
      }
      offset += hypoNum;
    }
  }
  if (offset != logprob.dims()[0]) {
    LOG(FATAL) << "Total number of hypos inconsistent : "
               << offset << " vs " << logprob.dims()[0];
  }
  return concatenate(outputVec, 0);
}

fl::Variable computeAdvantage(
    const fl::Variable& logprob, const std::vector<int>& hypoNums, 
    const double& margin) {
  // a(y) = logp(y) - min_{y' \in beam} logp(y') + margin
  std::vector<fl::Variable> out;
  int offset = 0;
  af::array baseline;
  fl::Variable logprobSlice;
  for (auto& hypoNum : hypoNums) {
    if (hypoNum == 0) {
      continue;
    }
    logprobSlice = logprob(af::seq(offset, offset+hypoNum-1));
    baseline = af::min(logprobSlice.array()) - margin;
    baseline = af::tile(baseline, af::dim4(hypoNum));
    out.emplace_back(logprobSlice - fl::noGrad(baseline));
    offset += hypoNum;
  }
  if (offset != logprob.dims()[0]) {
    LOG(FATAL) << "Total number of hypos inconsistent : "
               << offset << " vs " << logprob.dims()[0];
  }
  return concatenate(out, 0);
}

fl::Variable entropy(
    const fl::Variable& logprob, const std::vector<int>& hypoNums) {
  fl::Variable p = adjustProb(logprob, hypoNums, true, true);
  fl::Variable logp = fl::log(p + 1e-6);
  fl::Variable ent = fl::sum(fl::negate(p * logp), {0});
  ent = ent / static_cast<float>(hypoNums.size());
  return ent;
}

fl::Variable variableSum(
    const fl::Variable& var, const std::vector<int>& sizes, bool tile) {
  std::vector<fl::Variable> out;
  int offset = 0;
  for (auto& size : sizes) {
    fl::Variable val(af::constant(0, af::dim4(1)), false);
    if (size > 0) {
      val = fl::sum(var(af::seq(offset, offset+size-1)), {0});
    }

    if (!tile) {
      out.emplace_back(val);
    } else if (size > 0) {
      out.emplace_back(fl::tile(val, af::dim4(size)));
    }
    offset += size;
  }
  if (offset != var.dims()[0]) {
    LOG(FATAL) << "Total number of sizes inconsistent : "
               << offset << " vs " << var.dims()[0];
  }
  return concatenate(out, 0);
}

fl::Variable variableMax(
    const fl::Variable& var, const std::vector<int>& sizes, bool tile) {
  std::vector<fl::Variable> out;
  int offset = 0;
  for (auto& size : sizes) {
    fl::Variable val(af::constant(0, af::dim4(1)), false);
    if (size > 0) {
      auto varSlice = var(af::seq(offset, offset+size-1));
      af::array valArr, idxArr;
      af::max(valArr, idxArr, varSlice.array(), 0);
      val = varSlice(idxArr);
    }

    if (!tile) {
      out.emplace_back(val);
    } else if (size > 0) {
      out.emplace_back(fl::tile(val, af::dim4(size)));
    }
    offset += size;
  }
  if (offset != var.dims()[0]) {
    LOG(FATAL) << "Total number of sizes inconsistent : "
               << offset << " vs " << var.dims()[0];
  }
  return concatenate(out, 0);
}

fl::Variable computePriorMatchingLoss(
    const fl::Variable& lmLogprob, 
    const fl::Variable& s2sLogprob, 
    const std::vector<int>& hypoNums) {
  
  fl::Variable loss;
  if (FLAGS_pmType == kRegKL) {
    auto lmRenormProb = adjustProb(lmLogprob, hypoNums, true, true);
    loss = -1 * (lmRenormProb * s2sLogprob);
    loss = variableSum(loss, hypoNums);
  } else if (FLAGS_pmType == kRevKL) {
    auto advantage = computeAdvantage(lmLogprob, hypoNums, FLAGS_advmargin);
    loss = -1 * (fl::exp(s2sLogprob) * advantage);
    loss = variableSum(loss, hypoNums);
  } else {
    LOG(FATAL) << "Unsupported pmType : " << FLAGS_pmType;
  }
  return loss;
}

af::array batchTarget(
    const std::vector<std::vector<int>>& tgt, const int& padVal) {
  std::vector<int> vecTgt;
  af::dim4 vecTgtDims;
  int batchSz = tgt.size();
  size_t maxTgtSize = 0;

  for (const auto& t : tgt) {
    if (t.size() == 0) {
      LOG(FATAL) << "Target has zero length.";
    }
    maxTgtSize = std::max(maxTgtSize, t.size());
  }
  // L X BATCHSZ (Col Major)
  vecTgt.resize(maxTgtSize * batchSz, padVal);
  vecTgtDims = af::dim4(maxTgtSize, batchSz);
  
  for (size_t i = 0; i < batchSz; ++i) {
    std::copy(
        tgt[i].begin(),
        tgt[i].end(),
        vecTgt.begin() + maxTgtSize * i);
  }
  return af::array(vecTgtDims, vecTgt.data());
}

af::array makeOnehot(af::array& idx, const int& nClass) {
  int tgtLen = idx.dims()[0];
  int batchSz = idx.dims()[1];
  auto flatIdx = af::flat(idx);
  
  af::array A = af::range(af::dim4(nClass, tgtLen * batchSz));
  af::array B = af::reorder(flatIdx, 1, 0);  // [1, tgtLen * batchSz]
  B = af::tile(B, af::dim4(nClass, 1));
  auto onehot = (A == B).as(af::dtype::f32); // [nClass, tgtLen * batchSz]
  return af::moddims(onehot, af::dim4(nClass, tgtLen, batchSz));
}

fl::Variable sampleFromLogits(const fl::Variable& logits) {
  // logits : [bs, nclass], y : [bs]
  fl::Variable prob, cumprob, utri, sample, y;
  int nclass = logits.dims()[1];
  
  prob = fl::softmax(fl::noGrad(logits.array()), 1);
  utri = fl::noGrad(af::upper(af::constant(1, nclass, nclass)));
  cumprob = fl::matmul(prob, utri);
  cumprob = cumprob(af::span, af::seq(nclass - 1));
  sample = fl::tile(fl::noGrad(af::randu(cumprob.dims()[0])), {1, nclass - 1});
  y = fl::sum(sample > cumprob, {1});
  return y;
}

} // namespace w2l
