/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <fstream>
#include <stdexcept>

#include "experimental/semisupervised/src/runtime/Utils.h"

#include "common/Defines.h"
#include "common/Utils-base.h"
#include "experimental/semisupervised/src/runtime/Defines.h"

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

} // namespace w2l
