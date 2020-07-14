#pragma once

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <limits>

#include "libraries/common/Dictionary.h"


#include <flashlight/flashlight.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <vector>
#include "experimental/lead2Gold/src/criterion/l2g/NoiseLM.h"
#include "experimental/lead2Gold/src/data/NoiseW2lListFilesDataset.h"
#include "libraries/common/Dictionary.h"
#include "runtime/SpeechStatMeter.h"




//using namespace w2l;


class NoiseLMLetterSwapUnit : public NoiseLM {
protected:
  std::vector<fl::Variable> params_;
  std::vector<fl::Variable> temp_params_;

public:
  //NoiseLMLetterSwapUnit(int64_t nb_id_noise, int64_t nb_id_clean);
  explicit NoiseLMLetterSwapUnit(const std::string& FLAGS_probasdir, const std::string& FLAGS_noiselmtype, w2l::Dictionary& noise_keys, bool allowSwap, bool allowInsertion, bool allowDeletion,
                        bool autoScale, double scale_noise, double scale_sub, double scale_ins, double scale_del, double tkn_score = 0);
  double scoreSwap(int64_t id_noise, int64_t id_clean);
  double scoreSwap(const std::string& key_noise, const std::string& key_clean);
  double scoreInsertion(int64_t id_clean);
  double scoreInsertion(const std::string& key_clean);
  double scoreNoInsertion();
  double scoreDeletion(int64_t id_noise);
  double scoreDeletion(const std::string& key_noise);

  std::vector<fl::Variable> params() const {return params_;};
  std::vector<fl::Variable> temp_params() const {return temp_params_;};
  bool allowInsertion() {return allowInsertion_;};
  bool allowDeletion() {return allowDeletion_;};
  bool allowSwap() {return allowSwap_;};
  const std::string&  noiselmtype() {return noiselmtype_;};
  bool autoScale() {return autoScale_;};
  double scale_noise() {return scale_noise_;};
  double scale_sub() {return scale_sub_;};
  double scale_ins() {return scale_ins_;};
  double scale_del() {return scale_del_;};
  double tkn_score() {return tkn_score_;};
  void update_scale_noise(double scale_noise);
  //void zeroGrad(){};
  void accGradScore(int64_t sid, double gscore){};

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override {return inputs;}

  void setParams(const fl::Variable& var, int position) override {};

  std::string prettyString() const override {
    return "NoiseLMLetterSwapUnit";
  }
  
  void displayNoiseModel(bool logmode = false);

  void initialize();
  void update(af::array& pred, af::array& target, w2l::DictionaryMap& dicts, int replabel);
  //void finalize() override {};
  void finalize(bool enable_distributed);

  void trainModel(std::shared_ptr<w2l::W2lDataset> ds, std::shared_ptr<fl::Module> network, std::shared_ptr<w2l::SequenceCriterion> criterion, w2l::DictionaryMap& dicts, bool enable_distributed, int replabel, w2l::SpeechStatMeter& statsUnpaired);
  std::vector<double> NoiseDist(af::array& str1_af, af::array& str2_af,  w2l::DictionaryMap& dicts, int replabel);

  void evalBatch(af::array& str1_af, af::array& str2_af, w2l::DictionaryMap& dicts, int replabel, fl::AverageValueMeter& mtr);
  void evalModel(std::shared_ptr<fl::Module> ntwrk, std::shared_ptr<w2l::SequenceCriterion> criterion, int replabel, std::shared_ptr<w2l::W2lDataset> testds, w2l::DictionaryMap& dicts, fl::AverageValueMeter& mtr);
  void scaleToCpu();
  void paramsToCpu();
  void backward(){};
  ~NoiseLMLetterSwapUnit();

private:
  std::vector< std::vector<double> > logProbSwap;
  std::vector<double> logProbIns;
  std::vector<double> logProbDel;
  w2l::Dictionary noise_keys_;
  bool allowInsertion_;
  bool allowDeletion_;
  bool allowSwap_;
  const std::string& noiselmtype_;
  int64_t nb_key_;
  bool autoScale_;
  double scale_noise_, scale_sub_, scale_ins_, scale_del_, tkn_score_;

  std::vector< std::vector<int64_t> > countSwap_;
  std::vector<int64_t> countIns_;
  std::vector<int64_t> countDel_;
  int64_t nb_examples_;
  void rescale();

  //FL_SAVE_LOAD_WITH_BASE(
  //  fl::Module,
  //  noise_keys_,
  //  allowSwap_, allowInsertion_, allowDeletion_, 
  //  noiselmtype_, autoScale_,
  //  scale_subs_, scale_ins_, scale_del_,
  //  nb_key_,
  //  logProbSwap, logProbIns, logProbDel)

};

//CEREAL_REGISTER_TYPE(NoiseLMLetterSwapUnit)
