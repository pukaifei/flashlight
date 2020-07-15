#ifndef NOISELM_INC
#define NOISELM_INC

#include <string.h> // memcmp
#include <iostream>
#include <memory>
#include <vector>

#include <flashlight/flashlight.h>

#include "experimental/lead2Gold/src/criterion/l2g/NoiseTrie.h"

/*
class NoiseLMUnit {
public:
  virtual int64_t start() = 0;
  virtual int64_t add(int64_t sid, int lmidx) = 0;
  virtual int64_t finish(int64_t sid) = 0;
  virtual bool compare(int64_t sid1, int64_t sid2, NoiseTrieLabel *word) = 0;

  virtual float score(int64_t sid) = 0;
  virtual void zeroGrad() = 0;
  virtual void accGradScore(int64_t sid, double gscore) = 0;

  virtual ~NoiseLMUnit() {};
};
*/

class NoiseLM : public fl::Module {
  // protected:
  //  std::vector<fl::Variable> params_;

 public:
  // virtual std::vector<std::shared_ptr<NoiseLMUnit>> units(int64_t B) = 0;
  // virtual void backward();
  // virtual void initialize() {};
  // virtual void update(af::array& pred, af::array& target) {};
  // virtual void finalize() {};
  // virtual std::vector<fl::Variable> params() const {return params_;};
  // virtual ~NoiseLM() {};
};

/*
class NoiseLMHistory {
  std::vector<int64_t> states_;
  int64_t history_size_;
  int64_t add_()
  {
    int64_t id = states_.size();
    states_.resize(states_.size()+history_size_);
    return id/history_size_;
  }

public:
  NoiseLMHistory() { };
  void clear(int64_t history_size)
  {
    states_.clear();
    history_size_ = history_size;
  };
  int64_t add()
  {
    int64_t id = add_();
    for(int64_t i = 0; i < history_size_; i++) {
      states_[id*history_size_+i] = -1;
    }
    return id;
  };
  int64_t add(int64_t pid, int64_t widx)
  {
    int64_t id = add_();
    states_[id*history_size_] = widx;
    for(int64_t i = 0; i < history_size_-1; i++) {
      states_[id*history_size_+i+1] = states_[pid*history_size_+i];
    }
    return id;
  };
  int64_t get(int64_t id, int64_t o)
  {
    if(o < 0 || o > history_size_) {
      throw std::invalid_argument("order is out of range");
    }
    return states_[id*history_size_+o];
  };
  bool compare(int64_t id1, int64_t id2)
  {
    return memcmp(&states_[id1*history_size_], &states_[id2*history_size_],
sizeof(int64_t)*history_size_) == 0;
  };
  void print(int64_t id)
  {
    for(int64_t i = 0; i < history_size_; i++) {
      std::cout << " " << states_[id*history_size_+i];
    }
    std::cout << std::endl;
  }
};
*/
#endif
