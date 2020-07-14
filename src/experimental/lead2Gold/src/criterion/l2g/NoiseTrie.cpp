/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <stdexcept>
#include <limits>
#include <fstream>
#include <iostream>
#include <regex>
#include "experimental/lead2Gold/src/criterion/l2g/NoiseTrie.h"
#include "experimental/lead2Gold/src/common/Defines.h"
#include "common/Transforms.h"

NoiseTrie::NoiseTrie(int nchildren, int rootidx, void (*printsubwordunit)(int))
{
  node_id_ = 0;
  trielabel_id_ = 0;
  nchildren_ = nchildren;
  root_ = newnode(nchildren, rootidx);
  printsubwordunit_ = printsubwordunit;
}

void NoiseTrie::checkempty()
{
  if(!root_) {
    throw std::invalid_argument("Trie is empty");
  }
}
// ok it is recursive, whatever it's research ;)
void NoiseTrie::apply(NoiseTrieNode *node, std::function< void(NoiseTrieNode*) > func)
{
  func(node);
  for(NoiseTrieNode *child : node->children()) {
    if(child) {
      apply(child, func);
    }
  }
}

NoiseTrieNode* NoiseTrie::root()
{
  checkempty();
  return root_;
}

NoiseTrieNode* NoiseTrie::newnode(int nchildren, int idx)
{
  if(node_id_ != buffer_.size()) {
    throw std::invalid_argument("WTF");
  }
  buffer_.push_back({nchildren, idx, node_id_++});
  return &buffer_.back();
}

NoiseTrieNode* NoiseTrie::insert(const std::vector<int>& indices, int lm)
{
  checkempty();
  NoiseTrieLabel label({trielabel_id_++, lm});
  NoiseTrieNode *node = root_;
  for(size_t i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if(!node->child(idx)) {
      node->child(idx) = newnode(nchildren_, idx);
    }    
    node = node->child(idx);
  }
  node->labels().push_back(label);
  return node;
}

NoiseTrieNode* NoiseTrie::search(const std::vector<int>& indices)
{
  checkempty();
  NoiseTrieNode *node = root_;
  for(size_t i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if(!node->child(idx))
      return nullptr;
    node = node->child(idx);
  }
  return node;
}

void NoiseTrie::printsubwordunit(const int l)
{
  if(printsubwordunit_) {
    printsubwordunit_(l);
  } else {
    std::cout << l;
  }
}


NoiseTrieNode::NoiseTrieNode(int nchildren, int idx, long id)
{
  idx_ = idx;
  id_ = id;
  children_.resize(nchildren, nullptr);
}

void NoiseTrieNode::smearing(const std::vector<float>& word_scores, std::vector<float>& node_scores)
{
  node_scores.at(id_) = -std::numeric_limits<double>::infinity();
  for(NoiseTrieLabel& label : labels_) {
    node_scores.at(id_) = std::max(node_scores.at(id_), word_scores.at(label.lm));
  }
  for(auto child : children_) {
    if(child) {
      child->smearing(word_scores, node_scores); /* it is recursive */
      node_scores.at(id_) = std::max(node_scores.at(id_), node_scores.at(child->id_));
    }
  }
}

void NoiseTrie::smearing(const std::vector<float>& word_scores, std::vector<float>& node_scores)
{
  root_->smearing(word_scores, node_scores);
}

w2l::Dictionary NoiseTrie::load(const std::string filename, const w2l::Dictionary& tokens)
{
  w2l::Dictionary keys;
  std::ifstream f(filename);
  std::string line;
  std::regex re("(\\S+)");
  std::vector<int> indices;
  if(!f.good()) {
    throw std::invalid_argument("could not read file");
  }
  while(std::getline(f, line)) {
    indices.clear();
    std::sregex_iterator next(line.begin(), line.end(), re);
    std::sregex_iterator end;
    int kidx = -1;
    std::string key;
    while (next != end) {
      std::smatch match = *next;
      if(kidx < 0) {
        key = match.str();
        kidx = keys.indexSize();
      } else {
        indices.push_back(tokens.getIndex(match.str()));
      }
      next++;
    }
    //replaceReplabels(indices, w2l::FLAGS_replabel, tokens);
    insert(indices, kidx);
    keys.addEntry(key, kidx);
  }
  //int kidx = keys.indexSize();
  //keys.addEntry(w2l::kUnkToken, kidx);
  //keys.setDefaultIndex(kidx);
  return keys;
}

NoiseTrie::~NoiseTrie()
{
}


std::vector<NoiseTrieNode*>& NoiseTrieNode::children()
{
  return children_;
}

NoiseTrieNode*& NoiseTrieNode::child(int n)
{
  return children_.at(n);
}

std::deque<NoiseTrieLabel>& NoiseTrieNode::labels()
{
  return labels_;
}

NoiseTrieNode::~NoiseTrieNode()
{
}

