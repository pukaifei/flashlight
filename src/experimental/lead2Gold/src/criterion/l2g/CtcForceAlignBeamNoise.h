#ifndef CTCFULLCONNECTBEAMNOISE_INC
#define CTCFULLCONNECTBEAMNOISE_INC

#include "experimental/lead2Gold/src/criterion/l2g/NoiseLM.h"
#include "experimental/lead2Gold/src/criterion/l2g/NoiseLMLetterSwapPretrain.h"
#include "experimental/lead2Gold/src/criterion/l2g/NoiseTrie.h"
#include "libraries/common/Dictionary.h"
#include <map>


struct CtcForceAlignBeamNoiseNode
{
  long tok; // token index in tokenDict
  double score;
  double score_forsort;
  double logProbNoise;
  double maxscore;
  double gscore;
  int noisytarget_t;
  CtcForceAlignBeamNoiseNode* parent; // only for backtracking (Viterbi)
  //NoiseTrieNode *letter; 
  NoiseTrieNode *lexNode; 
  long lm;
  long merge_a;
  long merge_b;
  bool active;

  CtcForceAlignBeamNoiseNode()
    : tok(-1), score(0), score_forsort(0), logProbNoise(0), maxscore(0), noisytarget_t(-1), parent(nullptr), lexNode(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false)
  {
  }

  CtcForceAlignBeamNoiseNode(std::string mode)
    : tok(-1), score(-1e200), score_forsort(-1e200), logProbNoise(0), maxscore(0.), noisytarget_t(-1), parent(nullptr), lexNode(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false)
  {
  }


  CtcForceAlignBeamNoiseNode(long tok_, double score_, double score_forsort_, double tot_logProbNoise_, int noisytarget_t_, CtcForceAlignBeamNoiseNode *parent_, NoiseTrieNode *lexNode_, long lm_)
    : tok(tok_), score(score_), score_forsort(score_forsort_), logProbNoise(tot_logProbNoise_), maxscore(0.0), noisytarget_t(noisytarget_t_), parent(parent_), lexNode(lexNode_), lm(lm_), merge_a(-1), merge_b(-1), active(false)
  {
  }
  //ForceAlignBeamNoiseNode(ForceAlignBeamNoiseNode&) = delete; // deactivate copy constructor
  //ForceAlignBeamNoiseNode& operator=(ForceAlignBeamNoiseNode&&) = default;
};



struct CtcForceAlignBeamNoiseData
{
  std::vector< std::vector<CtcForceAlignBeamNoiseNode> > hyps; // use share pointeur // vector of hyp. 1 hyp is a vector of NoiseNode.
  std::vector<CtcForceAlignBeamNoiseNode*> merged; //use share pointeur
  std::vector<double> merged_score;
};

struct CtcForceAlignBeamNoiseBatchData // could use a vector only using ForceAlignBeamNoiseBatchData = std::vector<ForceAlignBeamNoiseData>;
{
  std::vector<CtcForceAlignBeamNoiseData> batch;
};

struct CtcForceAlignBeamNoiseVariablePayload : public fl::VariablePayload
{
  CtcForceAlignBeamNoiseVariablePayload(std::shared_ptr<CtcForceAlignBeamNoiseBatchData> data_) :
    data(std::move(data_)) {};
  std::shared_ptr<CtcForceAlignBeamNoiseBatchData> data;
  virtual ~CtcForceAlignBeamNoiseVariablePayload() = default;
};

class CtcForceAlignBeamNoise
{
  std::shared_ptr<NoiseTrie> lex_;
  NoiseLMLetterSwapUnit& noiselm_;
  w2l::Dictionary& tokenDict_;
  long B_;
  int top_k_;
  bool count_noise_, count_noise_sort_;
  int Nb_nested_;
  
  void backward(std::vector<fl::Variable>& inputs, const fl::Variable& goutput, std::shared_ptr<CtcForceAlignBeamNoiseBatchData> data);

public:

  double threshold_;
  double lengthPenality_;
  std::vector<double> stats_;

  CtcForceAlignBeamNoise(w2l::Dictionary& tokenDict, std::shared_ptr<NoiseTrie> lex, NoiseLMLetterSwapUnit& lm, long B, double threshold=0, int top_k=0, bool count_noise=true, bool count_noise_sort=false, int Nb_nested=1);
  fl::Variable forward(fl::Variable& emissions, fl::Variable& target);
  fl::Variable forward(fl::Variable& emissions, fl::Variable& emissions_forsort, fl::Variable& target);
  //fl::Variable transformOutput(fl::Variable& beam_previous, fl::Variable& emissions_previous, fl::Variable& emissions_new, fl::Variable& transitions, fl::Variable& target, fl::Variable& wtarget);
  static af::array viterbi(const fl::Variable& output);
  //static af::array viterbiWord(const fl::Variable& output);
  //static void showTarget(const fl::Variable& output, int64_t b, std::ostream& f); // debug tool
  //std::map< std::vector<int>, double> extractPathsAndWeights(const fl::Variable& output, int64_t b);
  //std::vector< std::tuple<std::vector<int>, double>> extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size);
  //double getTrueScore(const fl::Variable& output, int b, std::vector<int> pathAnalysis);
  //std::tuple<double, std::vector<std::vector< std::tuple<std::vector<int>, double>>> > wLER(const fl::Variable& output, fl::Variable& cleantarget, int beam_size, fl::AverageValueMeter* mtr_wLER = NULL);
  NoiseLMLetterSwapUnit& getNoiselm() {return noiselm_;};
  //void clearStats();
  //void statsAccDelta(double delta);
  //void statsAccMaxScore(double maxscore);
  //std::vector<double>& stats();
  ~CtcForceAlignBeamNoise() = default;
};

#endif
