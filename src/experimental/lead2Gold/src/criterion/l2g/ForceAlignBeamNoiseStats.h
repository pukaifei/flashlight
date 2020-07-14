#ifndef FULLCONNECTBEAMNOISESTATS_INC
#define FULLCONNECTBEAMNOISESTATS_INC

#include "experimental/lead2Gold/src/criterion/l2g/NoiseLM.h"
#include "experimental/lead2Gold/src/criterion/l2g/NoiseLMLetterSwapPretrain.h"
#include "experimental/lead2Gold/src/criterion/l2g/NoiseTrie.h"
#include "libraries/common/Dictionary.h"
#include <map>
#include <iomanip>

struct ForceAlignBeamNoiseNodeStats
{
  long l;
  double score;
  double maxscore;
  double cumul_maxscore;
  double gscore;
  int noisytarget_t;
  int knoisytarget_t;
  ForceAlignBeamNoiseNodeStats* parent; //shared pointeur // only for backtracking (Viterbi)
  NoiseTrieNode *letter; // shared pointeur
  NoiseTrieLabel *key; // share pointeur
  long lm;
  long merge_a;
  long merge_b;
  bool active;

  double length_trans;
  double nb_target_used;
  //std::vector<std::vector<double>> sub_token = std::vector<std::vector<double>>(28, std::vector<double>(28));
  //std::vector<double> del_token = std::vector<double>(28);
  //std::vector<double> ins_token = std::vector<double>(28);
  std::vector<std::vector<double>> sub_token;
  std::vector<double> del_token;
  std::vector<double> ins_token;

  double w_length_trans;
  double w_nb_target_used;
  //std::vector<std::vector<double>> w_sub_token = std::vector<std::vector<double>>(28, std::vector<double>(28));
  //std::vector<double> w_del_token = std::vector<double>(28);
  //std::vector<double> w_ins_token = std::vector<double>(28);
  std::vector<std::vector<double>> w_sub_token;
  std::vector<double> w_del_token;
  std::vector<double> w_ins_token;

  int nb_token;
  double sum_coeff;
  int target_size;
  int current_frame;
  int tot_frame;

  //~ForceAlignBeamNoiseNodeStats()
  //{
  //  std::cout << this << " is being destructed" << std::endl;
  //  delete length_trans_ptr, nb_target_used_ptr, sub_token_ptr, del_token_ptr, ins_token_ptr, w_length_trans_ptr, w_nb_target_used_ptr, w_sub_token_ptr, w_del_token_ptr, w_ins_token_ptr;
  //}
  
  ForceAlignBeamNoiseNodeStats()
  :l(-1), score(0), maxscore(0), cumul_maxscore(0.0), noisytarget_t(-1), knoisytarget_t(-1), parent(nullptr), letter(nullptr), key(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false), sum_coeff(1.)
  {

  }

  ForceAlignBeamNoiseNodeStats(std::string mode)
    : l(-1), score(-1e200), maxscore(0.), cumul_maxscore(0.0), noisytarget_t(-1), knoisytarget_t(-1), parent(nullptr), letter(nullptr), key(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false), sum_coeff(1.)
  {
  }


  ForceAlignBeamNoiseNodeStats(long l_, double score_, int noisytarget_t_, int knoisytarget_t_, ForceAlignBeamNoiseNodeStats *parent_, NoiseTrieNode *letter_, NoiseTrieLabel *key_,
                               long lm_, int nb_token_, int target_size_, int current_frame_, int tot_frame_)
    : l(l_), score(score_), maxscore(0.0), cumul_maxscore(0.0), noisytarget_t(noisytarget_t_), knoisytarget_t(knoisytarget_t_), parent(parent_), letter(letter_), key(key_),
     lm(lm_), merge_a(-1), merge_b(-1), active(false), nb_token(nb_token_), sum_coeff(1.), target_size(target_size_), current_frame(current_frame_), tot_frame(tot_frame_)
  {
    length_trans = 0.;
    nb_target_used = 0.;
    sub_token = std::vector<std::vector<double>>(nb_token, std::vector<double>(nb_token, 0.));
    del_token = std::vector<double>(nb_token, 0.);
    ins_token = std::vector<double>(nb_token, 0.);

    w_length_trans = 0.;
    w_nb_target_used = 0.;
    w_sub_token = std::vector<std::vector<double>>(nb_token, std::vector<double>(nb_token, 0.));
    w_del_token = std::vector<double>(nb_token, 0.);
    w_ins_token = std::vector<double>(nb_token, 0.);
  }

  ForceAlignBeamNoiseNodeStats(long l_, double score_, int noisytarget_t_, int knoisytarget_t_, ForceAlignBeamNoiseNodeStats *parent_, NoiseTrieNode *letter_, NoiseTrieLabel *key_, long lm_, int current_frame_)
    : l(l_), score(score_), maxscore(0.0), cumul_maxscore(0.0), noisytarget_t(noisytarget_t_), knoisytarget_t(knoisytarget_t_), parent(parent_), letter(letter_), key(key_),
     lm(lm_), merge_a(-1), merge_b(-1), active(false), nb_token(parent_->nb_token), sum_coeff(1.), target_size(parent_->target_size), current_frame(current_frame_), tot_frame(parent_->tot_frame),
    length_trans(parent_->length_trans),
    nb_target_used(parent_->nb_target_used),
    sub_token(parent_->sub_token),
    del_token(parent_->del_token),
    ins_token(parent_->ins_token),
    w_length_trans(parent_->w_length_trans),
    w_nb_target_used(parent_->w_nb_target_used),
    w_sub_token(parent_->w_sub_token),
    w_del_token(parent_->w_del_token),
    w_ins_token(parent_->w_ins_token)
  {
  }
  
  void set(long l_, double score_, int noisytarget_t_, int knoisytarget_t_, ForceAlignBeamNoiseNodeStats *parent_, NoiseTrieNode *letter_, NoiseTrieLabel *key_, long lm_, int current_frame_)
  {
    l = l_;
    score = score_;
    noisytarget_t = noisytarget_t_;
    knoisytarget_t = knoisytarget_t_;
    parent = parent_;
    letter = letter_;
    key = key_;
    lm = lm_;
    nb_token = parent_->nb_token;
    current_frame = current_frame;
    target_size = parent_->target_size;
    tot_frame = parent_->tot_frame;

    length_trans = parent_->length_trans;
    nb_target_used = parent_->nb_target_used;
    sub_token = parent_->sub_token;
    del_token = parent_->del_token;
    ins_token = parent_->ins_token;
    w_length_trans = parent_->w_length_trans;
    w_nb_target_used = parent_->w_nb_target_used;
    w_sub_token = parent_->w_sub_token;
    w_del_token = parent_->w_del_token;
    w_ins_token = parent_->w_ins_token;
    
    /*
    length_trans_ptr = std::make_shared<double>(*parent_->length_trans_ptr);
    nb_target_used_ptr = std::make_shared<double>(*parent_->nb_target_used_ptr);
    sub_token_ptr = std::make_shared<std::vector<std::vector<double>>>(*parent_->sub_token_ptr);
    del_token_ptr = std::make_shared<std::vector<double>>(*parent_->del_token_ptr);
    ins_token_ptr = std::make_shared<std::vector<double>>(*parent_->ins_token_ptr);

    w_length_trans_ptr = std::make_shared<double>(*parent_->w_length_trans_ptr);
    w_nb_target_used_ptr = std::make_shared<double>(*parent_->w_nb_target_used_ptr);
    w_sub_token_ptr = std::make_shared<std::vector<std::vector<double>>>(*parent_->w_sub_token_ptr);
    w_del_token_ptr = std::make_shared<std::vector<double>>(*parent_->w_del_token_ptr);
    w_ins_token_ptr = std::make_shared<std::vector<double>>(*parent_->w_ins_token_ptr);
    */
  }

  void updateSub(int noisy_, int clean_){

    length_trans += 1;
    w_length_trans += 1.;

    nb_target_used += 1;
    w_nb_target_used += 1.;

    sub_token[noisy_][clean_] += 1;
    w_sub_token[noisy_][clean_] += 1;
  }


  void updateIns(int noisy_){
    nb_target_used += 1;
    w_nb_target_used += 1.;

    ins_token[noisy_] += 1;
    w_ins_token[noisy_] += 1; 
  }

  void updateDel(int clean_){

    length_trans += 1;
    w_length_trans += 1;

    del_token[clean_] += 1;
    w_del_token[clean_] += 1;
  }

  void mergeWStats(ForceAlignBeamNoiseNodeStats& elem, double& coeff){

    w_length_trans += elem.w_length_trans * coeff;
    w_nb_target_used += elem.w_nb_target_used * coeff;

    for (int i=0 ; i<nb_token; i++){
      w_del_token[i] += elem.w_del_token[i] * coeff;
      w_ins_token[i] += elem.w_ins_token[i] * coeff;
      for (int j=0 ; j<nb_token; j++){
        w_sub_token[i][j] += elem.w_sub_token[i][j] * coeff;
      }
    }
    sum_coeff += coeff;
  }

  /*
  long l;
  double score;
  double maxscore;
  double gscore;
  int noisytarget_t;
  int knoisytarget_t;
  ForceAlignBeamNoiseNodeStats* parent; //shared pointeur // only for backtracking (Viterbi)
  NoiseTrieNode *letter; // shared pointeur
  NoiseTrieLabel *key; // share pointeur
  long lm;
  long merge_a;
  long merge_b;
  bool active;

  double *length_trans_ptr = nullptr;
  double *nb_target_used_ptr = nullptr;
  std::vector<std::vector<double>> *sub_token_ptr = nullptr;
  std::vector<double> *del_token_ptr = nullptr;
  std::vector<double> *ins_token_ptr = nullptr;

  double *w_length_trans_ptr = nullptr;
  double *w_nb_target_used_ptr = nullptr;
  std::vector<std::vector<double>> *w_sub_token_ptr = nullptr;
  std::vector<double> *w_del_token_ptr = nullptr;
  std::vector<double> *w_ins_token_ptr = nullptr;

  int nb_token;
  double sum_coeff;
  int target_size;
  int current_frame;
  int tot_frame;

  
  ForceAlignBeamNoiseNodeStats()
  :l(-1), score(0), maxscore(0), noisytarget_t(-1), knoisytarget_t(-1), parent(nullptr), letter(nullptr), key(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false), sum_coeff(1.)
  {
  }

  ForceAlignBeamNoiseNodeStats(std::string mode)
    : l(-1), score(-1e200), maxscore(-1), noisytarget_t(-1), knoisytarget_t(-1), parent(nullptr), letter(nullptr), key(nullptr), lm(-1), merge_a(-1), merge_b(-1), active(false), sum_coeff(1.)
  {
  }

  ForceAlignBeamNoiseNodeStats(long l_, double score_, int noisytarget_t_, int knoisytarget_t_, ForceAlignBeamNoiseNodeStats *parent_, NoiseTrieNode *letter_, NoiseTrieLabel *key_,
                               long lm_, int nb_token_, int target_size_, int current_frame_, int tot_frame_)
    : l(l_), score(score_), maxscore(0.0), noisytarget_t(noisytarget_t_), knoisytarget_t(knoisytarget_t_), parent(parent_), letter(letter_), key(key_),
     lm(lm_), merge_a(-1), merge_b(-1), active(false), nb_token(nb_token_), sum_coeff(1.), target_size(target_size_), current_frame(current_frame_), tot_frame(tot_frame_)
  {
    length_trans_ptr = new double(0.);
    nb_target_used_ptr = new double(0.);
    sub_token_ptr = new std::vector<std::vector<double>>(nb_token, std::vector<double>(nb_token, 0.));
    del_token_ptr = new std::vector<double>(nb_token, 0.);
    ins_token_ptr = new std::vector<double>(nb_token, 0.);

    w_length_trans_ptr = new double(0.);
    w_nb_target_used_ptr = new double(0.);
    w_sub_token_ptr = new std::vector<std::vector<double>>(nb_token, std::vector<double>(nb_token, 0.));
    w_del_token_ptr = new std::vector<double>(nb_token, 0.);
    w_ins_token_ptr = new std::vector<double>(nb_token, 0.);
  }
  
  ForceAlignBeamNoiseNodeStats(long l_, double score_, int noisytarget_t_, int knoisytarget_t_, ForceAlignBeamNoiseNodeStats *parent_, NoiseTrieNode *letter_, NoiseTrieLabel *key_, long lm_, int current_frame_)
    : l(l_), score(score_), maxscore(0.0), noisytarget_t(noisytarget_t_), knoisytarget_t(knoisytarget_t_), parent(parent_), letter(letter_), key(key_), lm(lm_), merge_a(-1), merge_b(-1), active(false), nb_token(parent_->nb_token), sum_coeff(1.),
      length_trans_ptr(parent_->length_trans_ptr), nb_target_used_ptr(parent_->nb_target_used_ptr), sub_token_ptr(parent_->sub_token_ptr), del_token_ptr(parent_->del_token_ptr), ins_token_ptr(parent_->ins_token_ptr),
      w_length_trans_ptr(parent_->w_length_trans_ptr), w_nb_target_used_ptr(parent_->w_nb_target_used_ptr), w_sub_token_ptr(parent_->w_sub_token_ptr), w_del_token_ptr(parent_->w_del_token_ptr), w_ins_token_ptr(parent_->w_ins_token_ptr),
      target_size(parent_->target_size), tot_frame(parent_->tot_frame), current_frame(current_frame_)
  {
  }

  void updateSub(int noisy_, int clean_){
    if (length_trans_ptr == parent->length_trans_ptr){
      length_trans_ptr = new double (*(parent->length_trans_ptr));
      w_length_trans_ptr = new double (*(parent->w_length_trans_ptr));
    }

    (*length_trans_ptr) += 1;
    (*w_length_trans_ptr) += 1.;

    if (nb_target_used_ptr == parent->nb_target_used_ptr){
      nb_target_used_ptr = new double (*(parent->nb_target_used_ptr));
      w_nb_target_used_ptr = new double (*(parent->w_nb_target_used_ptr));
    }

    (*nb_target_used_ptr) += 1;
    (*w_nb_target_used_ptr) += 1.;

    if (sub_token_ptr == parent->sub_token_ptr){
      sub_token_ptr = new std::vector<std::vector<double>> (*(parent->sub_token_ptr));
      w_sub_token_ptr = new std::vector<std::vector<double>> (*(parent->w_sub_token_ptr));
    }

    (*sub_token_ptr)[noisy_][clean_] += 1;
    (*w_sub_token_ptr)[noisy_][clean_] += 1;
  }


  void updateIns(int noisy_){
    if (nb_target_used_ptr == parent->nb_target_used_ptr){
      nb_target_used_ptr = new double (*(parent->nb_target_used_ptr));
      w_nb_target_used_ptr = new double (*(parent->w_nb_target_used_ptr));
    }

    (*nb_target_used_ptr) += 1;
    (*w_nb_target_used_ptr) += 1.;

    if (ins_token_ptr == parent->ins_token_ptr){
      ins_token_ptr = new std::vector<double> (*(parent->ins_token_ptr));
      w_ins_token_ptr = new std::vector<double> (*(parent->w_ins_token_ptr));
    }

    (*ins_token_ptr)[noisy_] += 1;
    (*w_ins_token_ptr)[noisy_] += 1; 
  }

  void updateDel(int clean_){
    if (length_trans_ptr == parent->length_trans_ptr){
      length_trans_ptr = new double (*(parent->length_trans_ptr));
      w_length_trans_ptr = new double (*(parent->w_length_trans_ptr));
    }

    (*length_trans_ptr) += 1;
    (*w_length_trans_ptr) += 1;

    if (del_token_ptr == parent->del_token_ptr){
      del_token_ptr = new std::vector<double> (*(parent->del_token_ptr));
      w_del_token_ptr = new std::vector<double> (*(parent->w_del_token_ptr));
    }

    (*del_token_ptr)[clean_] += 1;
    (*w_del_token_ptr)[clean_] += 1;
  }

  void mergeWStats(ForceAlignBeamNoiseNodeStats& elem, double& coeff){
    if (w_length_trans_ptr == parent->w_length_trans_ptr){
      w_length_trans_ptr = new double (*(parent->w_length_trans_ptr));
    }
    if (w_nb_target_used_ptr == parent->w_nb_target_used_ptr){
      w_nb_target_used_ptr = new double (*(parent->w_nb_target_used_ptr));
    }
    if (w_sub_token_ptr == parent->w_sub_token_ptr){
      w_sub_token_ptr = new std::vector<std::vector<double>> (*(parent->w_sub_token_ptr));
    }
    if (w_ins_token_ptr == parent->w_ins_token_ptr){
      w_ins_token_ptr = new std::vector<double> (*(parent->w_ins_token_ptr));
    }
    if (w_del_token_ptr == parent->w_del_token_ptr){
      w_del_token_ptr = new std::vector<double> (*(parent->w_del_token_ptr));
    }

    (*w_length_trans_ptr) += (*elem.w_length_trans_ptr) * coeff;
    (*w_nb_target_used_ptr) += (*elem.w_nb_target_used_ptr) * coeff;

    for (int i=0 ; i<nb_token; i++){
      (*w_del_token_ptr)[i] += (*elem.w_del_token_ptr)[i] * coeff;
      (*w_ins_token_ptr)[i] += (*elem.w_ins_token_ptr)[i] * coeff;
      for (int j=0 ; j<nb_token; j++){
        (*w_sub_token_ptr)[i][j] += (*elem.w_sub_token_ptr)[i][j] * coeff;
      }
    }
    sum_coeff += coeff;
  }
  */

  void finalizeWStats(){
    if (sum_coeff > 1) {    
      w_length_trans /= sum_coeff;
      w_nb_target_used /= sum_coeff;
      for (int i=0 ; i<nb_token; i++){
        w_del_token[i] /= sum_coeff;
        w_ins_token[i] /= sum_coeff;
        for (int j=0 ; j<nb_token; j++){
          w_sub_token[i][j] /= sum_coeff;
        }
      }
      sum_coeff = 1.0;
    }   
  }

  void addStatsFromOtherNode(ForceAlignBeamNoiseNodeStats& other){
    for (int i=0 ; i<nb_token; i++){
      ins_token[i] += other.ins_token[i];
      w_ins_token[i] += other.w_ins_token[i];

      del_token[i] += other.del_token[i];
      w_del_token[i] += other.w_del_token[i];

      for (int j=0 ; j<nb_token; j++){
        sub_token[i][j] += other.sub_token[i][j];
        w_sub_token[i][j] += other.w_sub_token[i][j];
      }
    }
    length_trans += other.length_trans;
    w_length_trans += other.w_length_trans;

    nb_target_used += other.nb_target_used;
    w_nb_target_used += other.w_nb_target_used;
    tot_frame += other.tot_frame;
    target_size += other.target_size;
    current_frame += other.current_frame;

  }


  std::vector<double> getEditStats(std::string mode = "max"){
    std::vector<std::vector<double>> sub;
    std::vector<double> ins;
    std::vector<double> del;
    double length;
    double target_used;
    if (mode == "max"){
      sub = sub_token;
      ins = ins_token;
      del = del_token;
      length = length_trans;
      target_used = nb_target_used;
    } else if (mode == "weighted"){
      sub = w_sub_token;
      ins = w_ins_token;
      del = w_del_token;
      length = w_length_trans;
      target_used = w_nb_target_used;
    } else {
      std::cout << "WRONG MODE" << std::endl;
      return {};
    }
    double count_sub=0;
    double count_ins=0;
    double count_del=0;
    for (int i=0 ; i<nb_token; i++){
        count_ins += ins[i];
        count_del += del[i];
      for (int j=0 ; j<nb_token; j++){
        if (i != j){
          count_sub += sub[i][j];
        }
      }
    }
    return {count_sub, count_ins, count_del, length, target_used};
  }

  void displayStats(std::string mode = "max"){
    if (mode != "max" && mode != "weighted"){
      std::cout << "WRONG MODE " << std::endl;
      return;
    }
    std::vector<double> editStats = getEditStats(mode);
    double count_sub = editStats[0];
    double count_ins = editStats[1];
    double count_del = editStats[2];
    double length = editStats[3];
    double target_used = editStats[4];

    std::cout << std::endl;
    if (length > 0) {
      std::cout << "mode " << mode << ", length " << length << std::endl;
      double norm_sub = count_sub / length * 100;
      double norm_ins = count_ins / length * 100;
      double norm_del = count_del / length * 100;
      double ratio_target = target_used / target_size * 100;
      double ratio_frame = (double)current_frame / tot_frame * 100;
      double R = ratio_frame / ratio_target;
      std::cout << "Sub: " << count_sub << ", " << norm_sub << "%" << std::endl;
      std::cout << "Ins: " << count_ins << ", " << norm_ins << "%" << std::endl;
      std::cout << "Del: " << count_del << ", " << norm_del << "%" << std::endl;
      std::cout << "Nb target used: " << target_used << ", " << ratio_target << "%" << std::endl;
      std::cout << "Frame: " << current_frame << ", " << ratio_frame << "%" << std::endl;
      std::cout << "R f/t: " << R << std::endl;
    } else{
      std::cout << "Something went wrong nb tokens = " << (length) << std::endl;
    }
  }

  void displayMatrix(std::string mode, bool norm, bool withInsDel, w2l::Dictionary* keys = nullptr){
    int width=6;
    int prec=1;
    //std::cout << std::setw(width) << std::setprecision(prec) << std::fixed; // display option
    std::cout << std::endl  << "SUB matrix mode " << mode << ", norm: " << norm << std::endl;
    std::vector<double> count_col(nb_token + 1, 0.); //+1 for insertion column
    std::vector<std::vector<double>> sub;
    std::vector<double> ins;
    std::vector<double> del;
    if (mode == "max"){
      sub = sub_token;
      ins = ins_token;
      del = del_token;
    } else if (mode == "weighted"){
      sub = w_sub_token;
      ins = w_ins_token;
      del = w_del_token;
    } else {
      std::cout << "WRONG MODE" << std::endl;
      return;
    }

    double count;
    for (int j=0 ; j<nb_token; j++){
      count=0;
      for (int i=0 ; i<nb_token; i++){
        count += sub[i][j];
      }
      if (withInsDel){
        count += del[j];
      }
      count_col[j] = count;
    }

    if (withInsDel){
      count=0;
      for (int i=0 ; i<nb_token; i++){
        count += ins[i];
      }
      count_col[nb_token] = count;
    }


    std::cout << "   ";
    for (int i=0 ; i<nb_token; i++){
      if (keys != nullptr){
        std::cout << std::setw(width) << keys->getEntry(i);
      } else{
        std::cout << std::setw(width) << "";
      }
    }
    std::cout << std::endl;
    std::string fill (width*(nb_token+1) + 5, '-');
    std::cout << fill << std::endl;
    //std::cout << std::setw(width) << std::setprecision(prec) << std::fixed;
    double display;
    for (int i=0 ; i<nb_token; i++){
      if (keys != nullptr){
        std::cout << std::setw(1) << keys->getEntry(i);     
      } else{
        std::cout << std::setw(1) << "";
      }
      std::cout << " |";
      for (int j=0 ; j<nb_token; j++){
        if (norm && count_col[j] > 0){
          display = sub[i][j] / count_col[j] * 100;
        } else{
          display = sub[i][j];
        }
        std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
      }
      if (withInsDel){ //Ins column
        std::cout << " |";
        if (norm && count_col[nb_token] > 0){
          display = ins[i] / count_col[nb_token] * 100;
        } else{
          display = ins[i];
        }
        std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display; 
      }
      std::cout << std::endl;
    }

    std::cout << fill << std::endl;
    if (withInsDel){ // Del row
      std::cout << "   ";
      for (int i=0 ; i<nb_token; i++){
        if (norm && count_col[i] > 0){
          display = del[i] / count_col[i] * 100;
        } else{
          display = del[i];
        }
        std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
      }
      std::cout << std::endl;
    }

    /*
    std::cout << "count per column" << std::endl;
    double count_tot=0.;
    std::cout << "   ";
    for (int j=0 ; j<nb_token; j++){
      count_tot += count_col[j];
      std::cout << std::setw(width) << std::setprecision(prec) << count_col[j];
    }
    std::cout << std::endl << "tot: " << count_tot << std::endl;
    */
    std::cout << std::setw(0) << std::setprecision(6) << std::defaultfloat; //get back to default display option
  }

  //ForceAlignBeamNoiseNode(ForceAlignBeamNoiseNode&) = delete; // deactivate copy constructor
  //ForceAlignBeamNoiseNode& operator=(ForceAlignBeamNoiseNode&&) = default;
};

ForceAlignBeamNoiseNodeStats aggregateStatsNodes(std::vector<ForceAlignBeamNoiseNodeStats>&);

void displayNoiseModel(NoiseLMLetterSwapUnit& noiselm, bool withInsDel, w2l::Dictionary* keys);


struct ForceAlignBeamNoiseStatsData
{
  std::vector< std::vector<ForceAlignBeamNoiseNodeStats> > hyps; // use share pointeur // vector of hyp. 1 hyp is a vector of NoiseNode.
  std::vector<ForceAlignBeamNoiseNodeStats*> merged; //use share pointeur
  std::vector<double> merged_score;
};

struct ForceAlignBeamNoiseStatsDataPtr
{
  std::vector< std::vector< ForceAlignBeamNoiseNodeStats* > > hyps; // use share pointeur // vector of hyp. 1 hyp is a vector of NoiseNode.
  std::vector< ForceAlignBeamNoiseNodeStats* > merged; //use share pointeur
  std::vector<double> merged_score;
};

struct ForceAlignBeamNoiseStatsBatchData // could use a vector only using ForceAlignBeamNoiseBatchData = std::vector<ForceAlignBeamNoiseData>;
{
  std::vector<ForceAlignBeamNoiseStatsData> batch;
};

/*
struct ForceAlignBeamNoiseData
{
  std::vector< std::vector<ForceAlignBeamNoiseNode> > hyps; // use share pointeur // vector of hyp. 1 hyp is a vector of NoiseNode.
  std::vector<ForceAlignBeamNoiseNode*> merged; //use share pointeur
  std::vector<double> merged_score;
};


struct ForceAlignBeamNoiseBatchData // could use a vector only using ForceAlignBeamNoiseBatchData = std::vector<ForceAlignBeamNoiseData>;
{
  std::vector<ForceAlignBeamNoiseData> batch;
};
*/


struct ForceAlignBeamNoiseStatsVariablePayload : public fl::VariablePayload
{
  ForceAlignBeamNoiseStatsVariablePayload(std::shared_ptr<ForceAlignBeamNoiseStatsBatchData> data_) :
    data(std::move(data_)) {};
  std::shared_ptr<ForceAlignBeamNoiseStatsBatchData> data;
  virtual ~ForceAlignBeamNoiseStatsVariablePayload() = default;
};

class ForceAlignBeamNoiseStats
{
  std::shared_ptr<NoiseTrie> lex_;
  NoiseLMLetterSwapUnit& noiselm_;
  w2l::Dictionary& tokenDict_;
  long B_;
  int top_k_;
  
  void backward(std::vector<fl::Variable>& inputs, const fl::Variable& goutput, std::shared_ptr<ForceAlignBeamNoiseStatsBatchData> data, bool is_target);

public:

  double threshold_;
  double lengthPenality_;
  std::vector<double> stats_;

  ForceAlignBeamNoiseStats(w2l::Dictionary& tokenDict, std::shared_ptr<NoiseTrie> lex, NoiseLMLetterSwapUnit& lm, long B, double threshold=0, int top_k=0);
  fl::Variable forward(fl::Variable& emissions, fl::Variable& transitions, fl::Variable& target, fl::Variable& wtarget);
  static af::array viterbi(const fl::Variable& output);
  static af::array viterbiWord(const fl::Variable& output);
  static void showTarget(const fl::Variable& output, int64_t b, std::ostream& f); // debug tool
  std::vector< std::tuple<std::vector<int>, double>> extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size);
  double getTrueScore(const fl::Variable& output, int b, std::vector<int> pathAnalysis);
  double wLER(const fl::Variable& output, fl::Variable& cleantarget, int& beam_size);
  std::vector< std::tuple<std::vector<int>, double>> extractPathsAndWeightsBackwardSimplify(const fl::Variable& output, int b, int beam_size);
  std::vector< std::tuple<std::vector<int>, double>> extractPathsAndWeightsForward(const fl::Variable& output, int b, int beam_size);
  std::vector< std::tuple<std::vector<int>, double>> extractPathsAndWeightsBoth(const fl::Variable& output, int b, int beam_size);
  std::vector< std::tuple<std::vector<int>, double>> greedyPath(const fl::Variable& output, int b);
  void simplifyGraph(ForceAlignBeamNoiseStatsData& data_big, ForceAlignBeamNoiseStatsDataPtr& data_small);
  void clearStats();
  void statsAccDelta(double delta);
  void statsAccMaxScore(double maxscore);
  std::vector<double>& stats();
  ~ForceAlignBeamNoiseStats() = default;
};

#endif
