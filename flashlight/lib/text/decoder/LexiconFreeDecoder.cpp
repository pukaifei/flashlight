/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#include "flashlight/lib/text/decoder/LexiconFreeDecoder.h"

namespace fl {
namespace lib {
namespace text {

void LexiconFreeDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconFreeDecoderState>());

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(0.0, lm_->start(0), nullptr, sil_);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconFreeDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  // 提前扩展hyp_的存储空间
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconFreeDecoderState>());
    }
  }

  std::vector<size_t> idx(N);
  // Looping over all the frames
  // 把t时刻emissions中最大的beamSizeToken个值的index放到idx中
  // 即只遍历概率最大的beamSizeToken个
  for (int t = 0; t < T; t++) {
    std::iota(idx.begin(), idx.end(), 0);
    if (N > opt_.beamSizeToken) {
      std::partial_sort(
          idx.begin(),
          idx.begin() + opt_.beamSizeToken,
          idx.end(),
          [&t, &N, &emissions](const size_t& l, const size_t& r) {
            return emissions[t * N + l] > emissions[t * N + r];
          });
    }

    //清空了candidates_(保存当前时刻的beam)
    //清空candidates_,candidatePtrs_,初始化candidatesBestScore_为inf
    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);
    for (const LexiconFreeDecoderState& prevHyp : hyp_[startFrame + t]) {
      const int prevIdx = prevHyp.token;

      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        // 当前时刻，需要向下扩展的beamSizeToken个char
        double amScore = emissions[t * N + n];
        // if (nDecodedFrames_ + t > 0 &&
        //     opt_.criterionType == CriterionType::ASG) {
        //   amScore += transitions_[n * N + prevIdx];
        // }
        double score = prevHyp.score + emissions[t * N + n];
        if (n == sil_) {
          score += opt_.silScore;
        }

        if ((opt_.criterionType == CriterionType::ASG && n != prevIdx) ||
            (opt_.criterionType == CriterionType::CTC && n != blank_ &&
             (n != prevIdx || prevHyp.prevBlank))) {
            //返回lm的状态和score
          auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
          auto lmScore = lmStateScorePair.second;

          //score及其之后参数调用LexiconFreeDecoderState构造函数生成一个struct对象，push到candidates_中
          // 当前向下扩展的是不同于前面的char/前面是sil
          // text会变化，新增一个char
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              score + opt_.lmWeight * lmScore,  
              lmStateScorePair.first,
              &prevHyp,
              n, //当前char对应的index
              false, // prevBlank
              prevHyp.amScore + amScore,
              prevHyp.lmScore + lmScore);
        } else if (opt_.criterionType == CriterionType::CTC && n == blank_) {
            // 当前向下扩展的char是sil
            // text不变
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              score, //上次beam的概率+blank的声学概率，没有语言模型概率
              prevHyp.lmState,
              &prevHyp,
              n,
              true, // prevBlank
              prevHyp.amScore + amScore,
              prevHyp.lmScore);
        } else {
            // 当前向下扩展的是和beam最后一个相同的char(no sil)
            // text不变
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              score,
              prevHyp.lmState,
              &prevHyp,
              n,
              false, // prevBlank
              prevHyp.amScore + amScore,
              prevHyp.lmScore);
        }
      }
    } //t时刻的beam loop end

    // merge
    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[startFrame + t + 1], //放到t+1的位置
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  } //t时刻loop end
  nDecodedFrames_ += T;
}

void LexiconFreeDecoder::decodeEnd() {
  candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);
  for (const LexiconFreeDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const LMStatePtr& prevLmState = prevHyp.lmState;

    auto lmStateScorePair = lm_->finish(prevLmState);
    auto lmScore = lmStateScorePair.second;

    candidatesAdd(
        candidates_,
        candidatesBestScore_,
        opt_.beamThreshold,
        prevHyp.score + opt_.lmWeight * lmScore,
        lmStateScorePair.first,
        &prevHyp,
        sil_,
        false, // prevBlank
        prevHyp.amScore,
        prevHyp.lmScore + lmScore);
  }

  candidatesStore(
      candidates_,
      candidatePtrs_,
      hyp_[nDecodedFrames_ - nPrunedFrames_ + 1],
      opt_.beamSize,
      candidatesBestScore_ - opt_.beamThreshold,
      opt_.logAdd,
      true);
  ++nDecodedFrames_;
}

std::vector<DecodeResult> LexiconFreeDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconFreeDecoder::getBestHypothesis(int lookBack) const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconFreeDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);

  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconFreeDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconFreeDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconFreeDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const LexiconFreeDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);
  if (!bestNode) {
    return; // Not enough decoded frames to prune
  }

  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (2) Move things from back of hyp_ to front and normalize scores */
  pruneAndNormalize(hyp_, startFrame, lookBack);

  nPrunedFrames_ = nDecodedFrames_ - lookBack;
}
} // namespace text
} // namespace lib
} // namespace fl
