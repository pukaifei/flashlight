#ifndef NOISELM_ZERO_INC
#define NOISELM_ZERO_INC

#include "experimental/lead2Gold/src/criterion/l2g/NoiseLM.h"

/*
class NoiseLMZeroUnit : public NoiseLMUnit {
public:
  NoiseLMZeroUnit() {};
  virtual int64_t start() { return 0; };
  virtual int64_t add(int64_t sid, int lmidx) { return 0; };
  virtual int64_t finish(int64_t sid) { return 0; };
  virtual bool compare(int64_t sid1, int64_t sid2, NoiseTrieLabel *word) { return true; };
  virtual float score(int64_t sid) { return 0.; };
  virtual void zeroGrad() {};
  virtual void accGradScore(int64_t sid, double gscore) {};
  virtual ~NoiseLMZeroUnit() {};
};
*/
  
class NoiseLMZero : public NoiseLM
{
public:
  NoiseLMZero()
  {
  };

  virtual void backward()
  {
  }
  virtual std::vector<fl::Variable> params()
  {
    return std::vector<fl::Variable>();
  }
  virtual ~NoiseLMZero() {};
};

#endif
