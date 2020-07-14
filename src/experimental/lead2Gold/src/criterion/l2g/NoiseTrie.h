 /**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef NOISETRIE_INC
#define NOISETRIE_INC

#include <deque>
#include <vector>
#include <functional>

#include "libraries/common/Dictionary.h"

struct NoiseTrieLabel {
  long id;
  int lm; /* lm label */
};

class NoiseTrieNode {
private:
  std::vector<NoiseTrieNode*> children_; /* letters */
  int idx_; /* letter */
  std::deque<NoiseTrieLabel> labels_; /* labels */
  long id_; /* unique id of the node */
public:
  NoiseTrieNode(int nchildren, int idx, long id);
  int idx() { return idx_; };
  int id() { return id_; };
  std::deque<NoiseTrieLabel>& labels();
  std::vector<NoiseTrieNode*>& children();
  NoiseTrieNode*& child(int idx);
  void smearing(const std::vector<float>& word_scores, std::vector<float>& node_scores);
  ~NoiseTrieNode();
};

class NoiseTrie {
private:
  NoiseTrieNode *root_;
  std::deque<NoiseTrieNode> buffer_;
  int nchildren_;
  void (*printsubwordunit_)(int);
  NoiseTrieNode *newnode(int nchildren, int idx);
  long node_id_;
  long trielabel_id_;
  void checkempty();
public:
  NoiseTrie(int nchildren, int rootidx, void (*printsubwordunit)(int));
  w2l::Dictionary load(const std::string filename, const w2l::Dictionary& tokens);
  NoiseTrieNode* root();
  NoiseTrieNode* insert(const std::vector<int>& indices, int lm);
  NoiseTrieNode* search(const std::vector<int>& indices);
  void smearing(const std::vector<float>& word_scores, std::vector<float>& node_scores);
  void printsubwordunit(const int l);
  void printword(const int l);
  void apply(NoiseTrieNode *node, std::function< void(NoiseTrieNode*) > func);
  size_t nNode() {return buffer_.size(); };
  ~NoiseTrie();  
};

#endif
