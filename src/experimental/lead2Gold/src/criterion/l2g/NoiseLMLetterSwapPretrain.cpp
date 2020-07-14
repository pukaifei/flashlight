#include "experimental/lead2Gold/src/criterion/l2g/NoiseLMLetterSwapPretrain.h"
#include "common/FlashlightUtils.h"
#include "experimental/lead2Gold/src/data/Utils.h"
#include "experimental/lead2Gold/src/common/Defines.h"
#include "libraries/common/WordUtils.h"
#include <fstream>
#include <regex>
#include <iomanip>


//const double NEG_INF_CLOSE = -1e10;
const double NEG_INF_CLOSE =  -std::numeric_limits<double>::infinity();

NoiseLMLetterSwapUnit::NoiseLMLetterSwapUnit(const std::string& FLAGS_probasdir, const std::string& FLAGS_noiselmtype,
  w2l::Dictionary& noise_keys, bool allowSwap, bool allowInsertion, bool allowDeletion, bool autoScale,
  double scale_noise, double scale_sub, double scale_ins, double scale_del, double tkn_score) :
  noise_keys_(noise_keys), allowSwap_(allowSwap),  allowInsertion_(allowInsertion),
  allowDeletion_(allowDeletion), noiselmtype_(FLAGS_noiselmtype), autoScale_(autoScale),
  scale_noise_(scale_noise), scale_sub_(scale_sub), scale_ins_(scale_ins), scale_del_(scale_del), tkn_score_(tkn_score)
{
  nb_key_ = noise_keys.indexSize();
  if (allowSwap){
    logProbSwap.resize(nb_key_, std::vector<double>(nb_key_, NEG_INF_CLOSE));
  }
  if (allowInsertion){
    logProbIns.resize(nb_key_ + 1, NEG_INF_CLOSE); // +1 is to store the proba not to insert a letter.
  }
  if (allowDeletion){
    logProbDel.resize(nb_key_, NEG_INF_CLOSE);
  }
  

  if (FLAGS_noiselmtype == "zeronoiselm"){
    if (allowSwap){
      for(int64_t i = 0; i < nb_key_; i++) {
        for (int64_t j = 0; j < nb_key_; j++) {
            logProbSwap[i][j] = 0.0;
        }
      }
    }
    if (allowInsertion){
      for(int64_t i = 0; i < nb_key_ + 1; i++) {
        logProbIns[i] = 0.0;
      }
    }
    if (allowDeletion){
      for(int64_t i = 0; i < nb_key_; i++) {
        logProbDel[i] = 0.0;
      }
    }
  } else if (FLAGS_noiselmtype == "identitynoiselm"){
    if (allowSwap){
      logProbSwap.resize(nb_key_, std::vector<double>(nb_key_, -std::numeric_limits<double>::infinity()));
      for(int64_t i = 0; i < nb_key_; i++) {
          logProbSwap[i][i] = 0.0;
      }
    }
    if (allowInsertion){
      logProbIns.resize(nb_key_ + 1, -std::numeric_limits<double>::infinity());
      logProbIns[nb_key_] = 0.0; // log proba  not to insert a letter
    }

  } else {

    std::string base_filename = std::string(FLAGS_probasdir) + std::string(FLAGS_noiselmtype);
    std::string filename;
    double proba;
    std::regex re("[+-]?([0-9]*[.])?[0-9]+"); //regex for a float//double

    if (allowSwap) {
      filename = base_filename + std::string("_prob_swap.txt");
      std::ifstream f(filename);
      std::string line;
      if(!f.good()) {
        throw std::invalid_argument("could not read swap probas file");
      }
      int i = 0;
      while(std::getline(f, line)) { //for every line
        if (i >= nb_key_){
          throw std::invalid_argument("too much lines in the swap file");
        }
        std::sregex_iterator next(line.begin(), line.end(), re);
        std::sregex_iterator end;
        int j = 0;
        while (next != end) { // for every column
          if (j >= nb_key_){
            throw std::invalid_argument("too much columns in the swap file");
          }
          std::smatch match = *next;
          proba = std::stod( match.str() );
          if (proba <= 0.){
            logProbSwap[i][j] = NEG_INF_CLOSE;
          } else{
            //if (allowDeletion){
            //  logProbSwap[i][j] = std::log(proba * ((float)nb_key_ / ((float)nb_key_ + 1.)));
            //} else{
            logProbSwap[i][j] = std::log(proba);
            //}
          }
          next++;
          j++;
        }
        if (j < nb_key_){
            throw std::invalid_argument("not enough column in the swap file");
        }
        i++;
      }
      if (i < nb_key_){
        throw std::invalid_argument("not enough line in the swap file");
      }
    }

    if (allowInsertion){
      filename = base_filename + std::string("_prob_ins.txt");
      std::ifstream f(filename);
      std::string line;
      if(!f.good()) {
        throw std::invalid_argument("could not read insertion probas file");
      }
      int i = 0;
      while(std::getline(f, line)) { //for every line
        if (i >= 1){
          throw std::invalid_argument("too much lines in insertion proba file");
        }
        std::sregex_iterator next(line.begin(), line.end(), re);
        std::sregex_iterator end;
        int j = 0;
        while (next != end) { // for every column
          if (j >= nb_key_ + 1){
            throw std::invalid_argument("too much columns in insertion proba file");
          }
          std::smatch match = *next;
          proba = std::stod( match.str() );
          if (proba == 0.){
            logProbIns[j] = NEG_INF_CLOSE;
          } else{
            //if (j != nb_key_){
            //  logProbIns[j] = std::log(proba * ((float)nb_key_ / ((float)nb_key_ + 1.)));
            //} else {
            //  logProbIns[j] = std::log(proba * (1. / ((float)nb_key_ + 1.)));
            //}
            logProbIns[j] = std::log(proba);        
          }
          next++;
          j++;
        }
        if (j < nb_key_ + 1){
            throw std::invalid_argument("not enough column in the swap file");
        }
        i++;
      }
      if (i < 1){
        throw std::invalid_argument("not enough line in the swap file");
      }
    }

    if (allowDeletion){
      filename = base_filename + std::string("_prob_del.txt");
      std::ifstream f(filename);
      std::string line;
      if(!f.good()) {
        throw std::invalid_argument("could not read deletion probas file");
      }
      int i = 0;
      while(std::getline(f, line)) { //for every line
        if (i >= 1){
          throw std::invalid_argument("too much lines in deletion proba file");
        }
        std::sregex_iterator next(line.begin(), line.end(), re);
        std::sregex_iterator end;
        int j = 0;
        while (next != end) { // for every column
          if (j >= nb_key_){
            throw std::invalid_argument("too much columns in deletion proba file");
          }
          std::smatch match = *next;
          proba = std::stod( match.str() );
          if (proba == 0.){
            logProbDel[j] = NEG_INF_CLOSE;
          } else{
            //logProbDel[j] = std::log(proba * (1. / ((float)nb_key_ + 1.)));
            logProbDel[j] = std::log(proba);
          }
          next++;
          j++;
        }
        if (j < nb_key_){
          throw std::invalid_argument("not enough column in the swap file");
        }
        i++;
      }
      if (i < 1){
        throw std::invalid_argument("not enough line in the swap file");
      }
    }

    //We add a correction to the probabilities
    //
    if (allowDeletion){
      for(int j = 0; j < nb_key_; j++) {
        //double log_norm_value = std::log(1 + std::exp(logProbDel[j]) * (scale_del_ - 1));

        double log_norm_value = scale_del_ * std::exp(logProbDel[j]);
        for (int i = 0; i < nb_key_; i++) {
          if (i == j){
            log_norm_value += std::exp(logProbSwap[j][j]);
          } else{
            log_norm_value += scale_sub_ * std::exp(logProbSwap[i][j]);
          }
        }
        log_norm_value = std::log(log_norm_value);

        for (int i = 0; i < nb_key_; i++) {
          if (i == j) {
            logProbSwap[j][j] -= log_norm_value;
          } else{
            logProbSwap[i][j] = std::log(scale_sub_) + logProbSwap[i][j] - log_norm_value;
          }
        }

        logProbDel[j] = std::log(scale_del_) + logProbDel[j] - log_norm_value;
      }
    } else{
      for(int j = 0; j < nb_key_; j++) {
        double log_norm_value = std::log(std::exp(logProbSwap[j][j]) * (1 - scale_sub_) + scale_sub_);

        for (int i = 0; i < nb_key_; i++) {
          if (i == j) {
            logProbSwap[j][j] -= log_norm_value;
          } else{
            logProbSwap[i][j] = std::log(scale_sub_) + logProbSwap[i][j] - log_norm_value;
          }
        }
      }
    }

    if (allowInsertion){
      double log_norm_value = std::log(1 + std::exp(logProbIns[nb_key_]) * (1./scale_ins_ - 1));
      for(int i = 0; i < nb_key_; i++) {
        logProbIns[i] -= log_norm_value;
      }
      logProbIns[nb_key_] = std::log(1./scale_ins_) + logProbIns[nb_key_] - log_norm_value;
    }
  }
  rescale();
  // Add to params
  auto logProbSwap_flat = std::vector<double>(nb_key_*nb_key_);
  for (int i = 0; i < nb_key_; i++) {
    std::copy(logProbSwap[i].begin(), logProbSwap[i].begin() + nb_key_, logProbSwap_flat.begin() + i*nb_key_);
  }

  params_ = {fl::Variable(af::array(1, &scale_noise_), true),
             fl::Variable(af::array(nb_key_*nb_key_, logProbSwap_flat.data()), false)};
  if (allowInsertion){
    params_.push_back(fl::Variable(af::array(nb_key_+1, logProbIns.data()), false));
  } else{
    params_.push_back(fl::Variable());
  }
  if (allowDeletion){
    params_.push_back(fl::Variable(af::array(nb_key_, logProbDel.data()), false));
  } else{
    params_.push_back(fl::Variable());
  }
}

void NoiseLMLetterSwapUnit::rescale(){
  //We add a correction to the probabilities
  if (allowDeletion_){
    for(int j = 0; j < nb_key_; j++) {
      double log_norm_value = scale_del_ * std::exp(logProbDel[j]);
      for (int i = 0; i < nb_key_; i++) {
        if (i == j){
          log_norm_value += std::exp(logProbSwap[j][j]);
        } else{
          log_norm_value += scale_sub_ * std::exp(logProbSwap[i][j]);
        }
      }
      log_norm_value = std::log(log_norm_value);

      for (int i = 0; i < nb_key_; i++) {
        if (i == j) {
          logProbSwap[j][j] -= log_norm_value;
        } else{
          logProbSwap[i][j] = std::log(scale_sub_) + logProbSwap[i][j] - log_norm_value;
        }
      }

      logProbDel[j] = std::log(scale_del_) + logProbDel[j] - log_norm_value;
    }
  } else{
    if (scale_sub_ != 1){
      for(int j = 0; j < nb_key_; j++) {
        double log_norm_value = std::log(std::exp(logProbSwap[j][j]) * (1 - scale_sub_) + scale_sub_);
        for (int i = 0; i < nb_key_; i++) {
          if (i == j) {
            logProbSwap[j][j] -= log_norm_value;
          } else{
            logProbSwap[i][j] = std::log(scale_sub_) + logProbSwap[i][j] - log_norm_value;
          }
        }
      }
    }
  }

  if (allowInsertion_){
    if (scale_ins_ != 1){
      double log_norm_value = std::log(1 + std::exp(logProbIns[nb_key_]) * (1./scale_ins_ - 1));
      for(int i = 0; i < nb_key_; i++) {
        logProbIns[i] -= log_norm_value;
      }
      logProbIns[nb_key_] = std::log(1./scale_ins_) + logProbIns[nb_key_] - log_norm_value;
    }
  }
}



NoiseLMLetterSwapUnit::~NoiseLMLetterSwapUnit()
{
}

double NoiseLMLetterSwapUnit::scoreSwap(const std::string& key_noise, const std::string& key_clean)
{
  if (allowSwap_){
    if (!noise_keys_.contains(key_noise) || !noise_keys_.contains(key_clean)){
      std::cout << "Error NoiseLMLetterSwapUnit::score, key_noise or key_clean is not a key" << std::endl;
    }
    auto id_noise = noise_keys_.getIndex(key_noise);
    auto id_clean = noise_keys_.getIndex(key_clean);
    //if (allowDeletion_){
    //  return scale_noise_ * (logProbSwap[id_noise][id_clean] + std::log(1 - std::exp(logProbDel[id_clean])));
    //} else{
    return logProbSwap[id_noise][id_clean];
    //}
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}

double NoiseLMLetterSwapUnit::scoreInsertion(const std::string& key_noise)
{
  if (allowInsertion_){
    if (!noise_keys_.contains(key_noise)){
      std::cout << "Error NoiseLMLetterSwapUnit::scoreInsertion, key_noise is not a key" << std::endl;
    }
    auto id_noise = noise_keys_.getIndex(key_noise);
    return logProbIns[id_noise];
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}


double NoiseLMLetterSwapUnit::scoreDeletion(const std::string& key_clean)
{
  if (allowDeletion_){
    if (!noise_keys_.contains(key_clean)){
      std::cout << "Error NoiseLMLetterSwapUnit::scoreDeletion, key_clean is not a key" << std::endl;
    }
    auto id_clean = noise_keys_.getIndex(key_clean);
    //return scale_noise_ * 2 * logProbDel[id_clean];
    return logProbDel[id_clean];
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}


double NoiseLMLetterSwapUnit::scoreSwap(int64_t id_noise, int64_t id_clean)
{
  //std::cout << "swap " << id_noise << " " << noise_keys_.getEntry(id_noise) << " " << id_clean << " " << noise_keys_.getEntry(id_clean) << " " << std::exp(logProbSwap[id_noise][id_clean]) << std::endl;
  //std::cout << "scale " << scale_subs_ << std::endl;
  if (allowSwap_){
    //std::cout << "swap " << scale_noise_ * logProbSwap[id_noise][id_clean] << std::endl;
    //if (allowDeletion_){
    //  return scale_noise_ * (logProbSwap[id_noise][id_clean] + std::log(1 - std::exp(logProbDel[id_clean])));
    //} else {
    return logProbSwap[id_noise][id_clean];
    //}

  } else {
    return -std::numeric_limits<double>::infinity();
  }
}

double NoiseLMLetterSwapUnit::scoreInsertion(int64_t id_noise)
{
  if (allowInsertion_){
    return logProbIns[id_noise];
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}

double NoiseLMLetterSwapUnit::scoreNoInsertion()
{
  if (allowInsertion_){
    return logProbIns[nb_key_];
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}

double NoiseLMLetterSwapUnit::scoreDeletion(int64_t id_clean)
{
  if (allowDeletion_){
    //return scale_noise_ * 2 * logProbDel[id_clean];
    return logProbDel[id_clean];
  } else {
    return -std::numeric_limits<double>::infinity();
  }
}

void NoiseLMLetterSwapUnit::update_scale_noise(double scale_noise)
{
  scale_noise_ = scale_noise;
}


void NoiseLMLetterSwapUnit::trainModel(
  std::shared_ptr<w2l::W2lDataset> ds,
  std::shared_ptr<fl::Module> network,
  std::shared_ptr<w2l::SequenceCriterion> criterion,
  w2l::DictionaryMap& dicts,
  bool enable_distributed,
  int replabel,
  w2l::SpeechStatMeter& statsUnpaired) {
  initialize();
  network->eval();
  nb_examples_ = 0;
  for (auto& sample : *ds) {
    statsUnpaired.add(sample[w2l::kInputIdx], sample[w2l::kTargetIdx]);
    auto output_eval = network->forward({fl::input(sample[w2l::kInputIdx])}).front();
    auto updatedTranscripts = getUpdateTrancripts(output_eval, criterion, dicts);
    
    for (int b = 0; b < sample[w2l::kInputIdx].dims(3); b++) {
      af::array cleanTarget = sample[w2l::kTargetIdx](af::span, b);
      af::array noisyTarget = updatedTranscripts[1](af::span, b);
      update(cleanTarget, noisyTarget, dicts, replabel);
    }
    nb_examples_ += sample[w2l::kInputIdx].dims(3);
  }
  finalize(enable_distributed);
  network->train();
}

void NoiseLMLetterSwapUnit::initialize() {
  if (allowSwap_){
    countSwap_.resize(nb_key_, std::vector<int64_t>(nb_key_, 0));
  }
  if (allowInsertion_){
    countIns_.resize(nb_key_ + 1, 0); // +1 is to store the proba not to insert a letter.
  }
  if (allowDeletion_){
    countDel_.resize(nb_key_, 0);
  }
};

struct editInfo {
  int l = 0; // Levenstein distance
  editInfo* prev = nullptr;
  int cleantkn = -1; // Describe operation 
  int noisytkn = -1; // Done
  void set(int l_, editInfo* prev_, int cleantkn_, int noisytkn_){
    l = l_;
    prev = prev_;
    cleantkn = cleantkn_;
    noisytkn = noisytkn_;
  }
  //~editInfo(){
    //if ( prev ) delete prev;
    //prev = nullptr;
  //}
};

void NoiseLMLetterSwapUnit::update(af::array& clean, af::array& noisy, w2l::DictionaryMap& dicts, int replabel) {
  auto cleanV = w2l::unpackReplabels(
                  w2l::afToVector<int>(clean),
                  dicts[w2l::kTargetIdx],
                  replabel
                );
      // Remove `-1`s appended to the target for batching (if any)
  auto cleanVlen = w2l::getTargetSize(cleanV.data(), cleanV.size());
  cleanV.resize(cleanVlen);

  auto noisyV = w2l::unpackReplabels(
                  w2l::afToVector<int>(noisy),
                  dicts[w2l::kTargetIdx],
                  replabel
                );
  auto noisyVlen = w2l::getTargetSize(noisyV.data(), noisyV.size());
  noisyV.resize(noisyVlen);

  int m = cleanV.size();
  int n = noisyV.size();
  if (n == 0){
      throw std::invalid_argument("Noisy transcription is empty");
  }
  // Create a table to store results of subproblems 
  editInfo dp[m + 1][n + 1]; // with dynamic programming

  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      if ((i == 0) and (j == 0))
        continue;
      // If first string is empty, only option is to 
      // insert all characters of second string 
      else if (i == 0) 
        dp[i][j].set(j, &dp[i][j - 1], -1, noisyV[j - 1]); // Insertion
      // If second string is empty, only option is to 
      // remove all characters of second string 
      else if (j == 0) 
        dp[i][j].set(i, &dp[i - 1][j], cleanV[i - 1], -1); // Deletion
  
      // If last characters are same, ignore last char 
      // and recur for remaining string 
      else if (cleanV[i - 1] == noisyV[j - 1]) 
        dp[i][j].set(dp[i - 1][j - 1].l, &dp[i - 1][j - 1], cleanV[i - 1], noisyV[j - 1]); // Replace tkn with the same dp[i - 1][j - 1]; 
  
      // If the last character is different, consider all 
      // possibilities and find the minimum 
      else {
        //If replace is the best one
        if  ((dp[i - 1][j - 1].l <= dp[i - 1][j].l) 
          && (dp[i - 1][j - 1].l <= dp[i][j - 1].l)){
          dp[i][j].set(1 + dp[i - 1][j - 1].l, &dp[i - 1][j - 1], cleanV[i - 1], noisyV[j - 1]);  
        //Else if insert is the best one
        } else if (dp[i][j - 1].l <= dp[i - 1][j].l){
          dp[i][j].set(1 + dp[i][j - 1].l, &dp[i][j - 1], -1, noisyV[j - 1]);
        // Else if Delete is the best one
        } else {
          dp[i][j].set(1 + dp[i - 1][j].l, &dp[i - 1][j], cleanV[i - 1], -1);
        }
      }
    }
  }
  // Now we get edit operations from the best alignment
  editInfo* current = &dp[m][n];
  int64_t tot_ins = 0;
  while (current->prev){
    // Find operation done during the current step
    if (current->cleantkn == -1){ // Insertion
      if (allowInsertion_)
        countIns_[current->noisytkn] += 1;
      tot_ins += 1;
    } else if (current->noisytkn == -1) { // Deletion
      if (allowDeletion_)
        countDel_[current->cleantkn] += 1; 
    } else {
      countSwap_[current->noisytkn][current->cleantkn] += 1;
    }
    current = current->prev;
  }
  if (allowInsertion_){ //Compute not to insert. Approximate number of time we can insert by m+1 
    int noIns = std::max(m + 1 - tot_ins, (int64_t)0);
    countIns_[nb_key_] += noIns;
  }
};

//Transform into fl Variable and add to params_, go to gpu, all reduce, back to cpu and normnalize
void NoiseLMLetterSwapUnit::finalize(bool enable_distributed){
  //fl::Variable(af::array(1, &scale_noise_), true)
  std::vector< std::vector<double> > countSwapNorm_;
  std::vector<double> countInsNorm_;
  std::vector<double> countDelNorm_;

  //We Norm by the number of examples before Allreduce. Avoid overflow.

  if (allowSwap_){
    countSwapNorm_.resize(nb_key_, std::vector<double>(nb_key_, 0));
    for (int i = 0; i < nb_key_; i++) {
      for (int j = 0; j < nb_key_; j++) {
        countSwapNorm_[i][j] = (double)countSwap_[i][j] / (double)nb_examples_;
        //std::cout << (double)countSwap_[i][j] << " " << (double)nb_examples_ << " swa-> " << countSwapNorm_[i][j] << std::endl;
      }
    }
  }
  if (allowInsertion_){
    countInsNorm_.resize(nb_key_ + 1, 0); // +1 is to store the proba not to insert a letter.
    for (int i = 0; i < nb_key_ + 1; i++){
      countInsNorm_[i] = (double)countIns_[i] / (double)nb_examples_;
      //std::cout << (double)countIns_[i] << " " << (double)nb_examples_ << " ins-> " << countInsNorm_[i] << std::endl;
    }
  }
  if (allowDeletion_){
    countDelNorm_.resize(nb_key_, 0);
    for (int j = 0; j < nb_key_ ; j++){
      countDelNorm_[j] = (double)countDel_[j] / (double)nb_examples_;
      //std::cout << (double)countDel_[j] << " " << (double)nb_examples_ << " del-> " << (double)countDelNorm_[j] << std::endl;
    }
  }


  if (enable_distributed){
    auto countSwap_flatNorm_ = std::vector<double>(nb_key_*nb_key_);
    for (int i = 0; i < nb_key_; i++) {
      std::copy(countSwapNorm_[i].begin(), countSwapNorm_[i].begin() + nb_key_, countSwap_flatNorm_.begin() + i*nb_key_);
    }


    temp_params_ = {fl::Variable(af::array(nb_key_*nb_key_, countSwap_flatNorm_.data()), false)};
    if (allowInsertion_){
      temp_params_.push_back(fl::Variable(af::array(nb_key_+1, countInsNorm_.data()), false));
    }
    if (allowDeletion_){
      temp_params_.push_back(fl::Variable(af::array(nb_key_, countDelNorm_.data()), false));
    }
    allReduceMultiple(temp_params_);

    //Unflat Swap
    countSwap_flatNorm_ = w2l::afToVector<double>(temp_params_[0]);

    for (int i = 0; i < nb_key_; i++) {
      std::copy(countSwap_flatNorm_.begin() + i*nb_key_,
                countSwap_flatNorm_.begin() + (i+1)*nb_key_,
                countSwapNorm_[i].begin());
    }


    if (allowInsertion_){
      countInsNorm_ = w2l::afToVector<double>(temp_params_[1]);
    }

    if (allowDeletion_){
      countDelNorm_ = w2l::afToVector<double>(temp_params_[2]);
    }   
  }

  //Now Compute log Prob
  
  for (int j=0; j < nb_key_ ; j++){
    double tkn_count_j = 0;
    for (int i=0; i < nb_key_; i++){
      //std::cout << "count swap " << i << " " << j << " " << countSwap_[i][j] << std::endl;
      tkn_count_j += countSwapNorm_[i][j];
    }
    if (allowDeletion_){
      tkn_count_j += countDelNorm_[j];
      //std::cout << "count del " << j << " " << countDel_[j] << std::endl;
      if (countDelNorm_[j] == 0 || tkn_count_j == 0){
        logProbDel[j] = NEG_INF_CLOSE;
      } else{
        logProbDel[j] = std::log(countDelNorm_[j] / tkn_count_j);
        //std::cout << "delnorm " << countDelNorm_[j] << " " << tkn_count_j << " -> " << logProbDel[j] <<  std::endl;
      }
    }

    for (int i=0; i < nb_key_; i++){
      if (countSwapNorm_[i][j] == 0 || tkn_count_j == 0){
        logProbSwap[i][j] = NEG_INF_CLOSE;
      } else{
        logProbSwap[i][j] = std::log(countSwapNorm_[i][j] / tkn_count_j);
        //std::cout << "swapnorm " << countSwapNorm_[i][j] << " " << tkn_count_j << " -> " << logProbSwap[i][j] << std::endl;
      }
    }
  }

  if (allowInsertion_){
    double tkn_count_ins = 0;
    for (int i=0; i < nb_key_ + 1; i++){
      //std::cout << "count ins " << i << " " << countIns_[i] << std::endl;
      tkn_count_ins += countInsNorm_[i];
    }
    for (int i=0; i < nb_key_ + 1; i++){
      if (countInsNorm_[i] == 0 || tkn_count_ins == 0){
        logProbIns[i] = NEG_INF_CLOSE;
      } else{
        logProbIns[i] = std::log(countInsNorm_[i] / tkn_count_ins);
        //std::cout << "insnorm " << countInsNorm_[i] << " " << tkn_count_ins << " -> " << logProbIns[i]  << std::endl;
      }
    }
  }
  rescale();

  // Finally go back to params
  auto logProbSwap_flat = std::vector<double>(nb_key_*nb_key_);
  for (int i = 0; i < nb_key_; i++) {
    std::copy(logProbSwap[i].begin(), logProbSwap[i].begin() + nb_key_, logProbSwap_flat.begin() + i*nb_key_);
  }
  params_[1] = fl::Variable(af::array(nb_key_*nb_key_, logProbSwap_flat.data()), false);

  if (allowInsertion_){
    params_[2] = fl::Variable(af::array(nb_key_+1, logProbIns.data()), false);
  }
  if (allowDeletion_){
    params_[3] = fl::Variable(af::array(nb_key_, logProbDel.data()), false);
  }
}

void NoiseLMLetterSwapUnit::scaleToCpu(){
  scale_noise_ = params_[0].scalar<double>();
}

void NoiseLMLetterSwapUnit::paramsToCpu(){
  scaleToCpu();

  auto logProbSwap_flat = w2l::afToVector<double>(params_[1]);
  for (int i = 0; i < nb_key_; i++) {
    std::copy(logProbSwap_flat.begin() + i*nb_key_,
              logProbSwap_flat.begin() + (i+1)*nb_key_,
              logProbSwap[i].begin());
  }
  if (allowInsertion_){
    logProbIns = w2l::afToVector<double>(params_[2]);
  }
  if (allowDeletion_){
    logProbDel = w2l::afToVector<double>(params_[3]);
  }
}

void NoiseLMLetterSwapUnit::displayNoiseModel(bool logmode){
  int width, prec;
  if (logmode){
    width=12;
    prec=4;
  } else{
    width=6;
    prec=1;
  }

  int nb_token = nb_key_;
  std::cout << std::endl  << "Noise model " << std::endl;
  std::cout << "   ";
  for (int i=0 ; i<nb_token; i++){
    std::cout << std::setw(width) << noise_keys_.getEntry(i);
  }

  std::cout << std::endl;
  std::string fill (width*(nb_token+1) + 5, '-');
  std::cout << fill << std::endl;
  double display;
  for (int i=0 ; i<nb_token; i++){
    std::cout << std::setw(1) << noise_keys_.getEntry(i);     
    std::cout << " |";
    for (int j=0 ; j<nb_token; j++){
      if (logmode)
        display = scoreSwap(i,j);
      else
        display = exp(scoreSwap(i,j)) * 100;
      std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
    }
    if (allowInsertion_){ //Ins column
      std::cout << " |";
      if (logmode)
        display = scoreInsertion(i);
      else
        display = exp(scoreInsertion(i)) * 100;
      std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display; 
    }
    std::cout << std::endl;
  }

  std::cout << fill << std::endl;
  if (allowDeletion_){ // Del row
    std::cout << "   ";
    for (int i=0 ; i<nb_token; i++){
      if (logmode)
        display = scoreDeletion(i);
      else
        display = exp(scoreDeletion(i)) * 100;
      std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
    }
    if (logmode)
      display = scoreNoInsertion();
    else
      display = exp(scoreNoInsertion()) * 100;
    std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
    std::cout << std::endl;
  }

  std::cout << std::setw(0) << std::setprecision(6) << std::defaultfloat; //get back to default display option
}


// !! Score No insertion not taken into accout
std::vector<double> NoiseLMLetterSwapUnit::NoiseDist(af::array& str1_af, af::array& str2_af,  w2l::DictionaryMap& dicts, int replabel) 
{

  auto str1 = w2l::unpackReplabels(
                  w2l::afToVector<int>(str1_af),
                  dicts[w2l::kTargetIdx],
                  replabel
                );

  auto str1len = w2l::getTargetSize(str1.data(), str1.size());
  str1.resize(str1len);

  auto str2 = w2l::unpackReplabels(
                  w2l::afToVector<int>(str2_af),
                  dicts[w2l::kTargetIdx],
                  replabel
                );
  auto str2len = w2l::getTargetSize(str2.data(), str2.size());
  str2.resize(str2len);
  //str1 clean to str2 noisy
  // Create a table to store results of subproblems

    int m = str1.size();
    int n = str2.size();
    if (n == 0){
      throw std::invalid_argument("Noisy transcription is empty");
    }
    //std::cout << "a";
    double dp_noise[m + 1][n + 1];
    //std::cout << " b";
    double dp_noise_ij_ins, dp_noise_ij_del, dp_noise_ij_sub;
  
    // Fill d[][] in bottom up manner 
    for (int i = 0; i <= m; i++) { 
      for (int j = 0; j <= n; j++) { 
        // If first string is empty, only option is to 
        // insert all characters of second string
        if (i == 0 && j==0){
          //std::cout << " c";
          dp_noise[i][j] = 0;
          //std::cout << " d";
        }
        
        else if (i == 0){ // ---> insertion first line
          //std::cout << " e";
          dp_noise[i][j] = dp_noise[i][j-1] + scoreInsertion(str2[j-1]);
          //std::cout << " f";       
        } 
        // If second string is empty, only option is to 
        // remove all characters of second string 
        else if (j == 0){
          //std::cout << " g";
          dp_noise[i][j] = dp_noise[i-1][j] + scoreDeletion(str1[i-1]);
          //std::cout << " h";
        }
        else{
          //std::cout << " i";
          dp_noise_ij_ins = dp_noise[i][j - 1] + scoreInsertion(str2[j-1]);
          //std::cout << " j";
          dp_noise_ij_del = dp_noise[i-1][j] + scoreDeletion(str1[i-1]);
          //std::cout << " k";
          dp_noise_ij_sub = dp_noise[i-1][j-1] + scoreSwap(str2[j-1], str1[i-1]);
          //std::cout << " l";

          if (dp_noise_ij_sub > dp_noise_ij_del && dp_noise_ij_sub > dp_noise_ij_ins){
            //std::cout << " k";
            dp_noise[i][j] = dp_noise_ij_sub;
            //std::cout << " l";
          } else{
            if (dp_noise_ij_del > dp_noise_ij_ins){
              //std::cout << " m";
              dp_noise[i][j] = dp_noise_ij_del;
              //std::cout << " n";
            } else{
              //std::cout << " o";
              dp_noise[i][j] = dp_noise_ij_ins;
              //std::cout << " p";
            }
          }
        }         
      } 
    }
  //std::cout << " q";
  //std::cout << " " << &dp_noise[m][n];
  //std::cout << " --> " << dp_noise[m][n];
  //std::cout << " r" << std::endl;
  return {dp_noise[m][n], std::exp(dp_noise[m][n] / m)};
}


void NoiseLMLetterSwapUnit::evalBatch(af::array& str1_af_batch, af::array& str2_af_batch, w2l::DictionaryMap& dicts, int replabel, fl::AverageValueMeter& mtr){
  for (int b=0 ; b < str1_af_batch.dims(1) ; b++){
    af::array str1_af = str1_af_batch(af::span,b);
    af::array str2_af = str2_af_batch(af::span,b);
    auto resEdit = NoiseDist(str1_af, str2_af, dicts, replabel);
    auto& p_noise_clean_b = resEdit[0];
    auto& p_noise_clean_norm_b = resEdit[1];
    if (p_noise_clean_b > -1e5){
        mtr.add(p_noise_clean_norm_b);
    }
  }
}

void NoiseLMLetterSwapUnit::evalModel(
  std::shared_ptr<fl::Module> ntwrk,
  std::shared_ptr<w2l::SequenceCriterion> criterion,
  int replabel,
  std::shared_ptr<w2l::W2lDataset> testds,
  w2l::DictionaryMap& dicts,
  fl::AverageValueMeter& mtr){
  ntwrk->eval();
  for (auto& sample : *testds) {
    auto output = ntwrk->forward({fl::input(sample[w2l::kInputIdx])}).front();
    std::vector<af::array> updatedTranscripts;
    updatedTranscripts = getUpdateTrancripts(output, criterion, dicts);
    evalBatch(sample[w2l::kTargetIdx], updatedTranscripts[1], dicts, replabel, mtr);
  }
  ntwrk->train();
}



