/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <utility>

#include <flashlight/flashlight.h>

#include "libraries/common/Dictionary.h"
#include "criterion/criterion.h"
#include "experimental/localPriorMatchOss/src/module/LMCritic.h"

namespace w2l {
Dictionary createFairseqTokenDict(const std::string& filepath);

/**
 * Generate mapping between the indices of tokens in dict1 and dict2
 * for matching the dictionaries in w2l and fairseq.
 * The function returns (mapping, numPadding), where the
 * token with index i in dict1 maps to the token with index mapping[i] in dict2.
 * numPadding is the number of tokens that appear in dict1 but not dict2,
 * and we map those tokens to dict2.indexSize() + 0, dict2.indexSize() + 1, ...
 * dict2.indexSize() + numPadding - 1 in order.
 */
std::pair<std::vector<int>, int> genTokenDictIndexMap(
    const Dictionary& dict1,
    const Dictionary& dict2);

void print_path(std::vector<int> path, Dictionary& dict);

std::vector<int> remapLabelsForLM(std::vector<int> path, Dictionary& dict);

std::string arrDimStr(const af::array& arr);

template <class T>
std::string stringify(const std::vector<T>& vec, std::string sep=" ");

template<class T, class S>
std::vector<S> getLengths(const std::vector<std::vector<T>>& vec);

af::array getTargetLength(af::array& target, int eosIdx);

std::pair<std::vector<std::vector<int>>, std::vector<int>> batchBeamSearch(
    const fl::Variable& output,
    const std::shared_ptr<Seq2SeqCriterion>& criterion,
    int eos);

std::pair<std::vector<std::vector<int>>, std::vector<int>> filterBeamByLength(
    const std::vector<std::vector<int>>& paths,
    const std::vector<int>& hypoNums,
    const std::vector<int>& refLengths);

fl::Variable computeLmLogprob(
    const std::vector<std::vector<int>>& paths,
    const std::shared_ptr<LMCritic>& lmcrit,
    const Dictionary& dict);

fl::Variable computeS2SLogprob(
    const std::vector<std::vector<int>>& paths,
    const std::vector<int>& hypoNums,
    const fl::Variable& encoderOutput,
    const std::shared_ptr<Seq2SeqCriterion>& criterion,
    const Dictionary& dict);

fl::Variable adjustProb(
    const fl::Variable& logprob, const std::vector<int>& hypoNums,
    bool renormalize, bool linear);

fl::Variable entropy(
    const fl::Variable& logprob, const std::vector<int>& hypoNums);

fl::Variable variableSum(
    const fl::Variable& var, const std::vector<int>& sizes, bool tile=false);

fl::Variable variableMax(
    const fl::Variable& var, const std::vector<int>& sizes, bool tile=false);

fl::Variable computePriorMatchingLoss(
    const fl::Variable& lmLogprob,
    const fl::Variable& s2sLogprob,
    const std::vector<int>& hypoNums);

af::array batchTarget(
    const std::vector<std::vector<int>>& tgt, const int& padVal);

af::array makeOnehot(af::array& idx, const int& nClass);

fl::Variable sampleFromLogits(const fl::Variable& logits);

} // namespace w2l

#include "experimental/localPriorMatchOss/src/runtime/Utils-inl.h"
