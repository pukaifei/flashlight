#include "experimental/lead2Gold/src/criterion/l2g/ForceAlignBeamNoiseStats.h"

// for w2l::gettargetSize()
#include "criterion/CriterionUtils.h"
#include <iomanip>
#include <algorithm>
#include <assert.h>
#include <tuple>
#include "common/Transforms.h"

//#include <fenv.h> //debug, target nan values




static double logadd(double a, double b)
{
  if (a == -std::numeric_limits<double>::infinity() && b == -std::numeric_limits<double>::infinity()){
    return a;
  } 
  if(a > b) {
    return a + log1p(exp(b-a));
  } 
  return b + log1p(exp(a-b));
}

static void printhypsfast(long t, std::vector<ForceAlignBeamNoiseNodeStats>& hyps, long nhyp)
{
  std::cout << "FST ===============================================================" << std::endl;
  double norm = 0;
  for(long h = 0; h < nhyp; h++) {
    ForceAlignBeamNoiseNodeStats& node = hyps[h];
    std::cout << "[" << h << "] t=" << t << " l=" << node.l << " score=" << node.score << " noisytarget_t=" << node.noisytarget_t << std::endl;
    norm += node.score*node.score;
  }
  std::cout << "norm=" << norm << std::endl;
}

ForceAlignBeamNoiseStats::ForceAlignBeamNoiseStats(w2l::Dictionary& tokenDict, std::shared_ptr<NoiseTrie> lex, NoiseLMLetterSwapUnit& noiselm, long B, double threshold, int top_k)
 : tokenDict_(tokenDict), lex_(lex), noiselm_(noiselm), B_(B), threshold_(threshold), top_k_(top_k)
{
}

void ForceAlignBeamNoiseStats::statsAccDelta(double delta)
{
  stats_.at(0) += 1;
  stats_.at(1) += fabs(delta);
  stats_.at(2) += fabs(delta)*fabs(delta);
}

void ForceAlignBeamNoiseStats::statsAccMaxScore(double maxscore)
{
  stats_.at(3) += 1;
  stats_.at(4) += fabs(maxscore);
}

void ForceAlignBeamNoiseStats::clearStats()
{
  stats_.resize(5);
  for(size_t i = 0; i < stats_.size(); i++) {
    stats_.at(i) = 0;
  }
}

std::vector<double>& ForceAlignBeamNoiseStats::stats()
{
  return stats_;
}

struct Comp{ // to sort idx for top k feature
    Comp( const float* v ) : _v(v) {}
    bool operator ()(int a, int b) { return _v[a] > _v[b]; }
    const float* _v;
};

fl::Variable ForceAlignBeamNoiseStats::forward(fl::Variable& emissions, fl::Variable& transitions, fl::Variable& noisytarget, fl::Variable& knoisytarget)
{
  //feenableexcept(FE_INVALID | FE_OVERFLOW); //the debugger will warn if nan
  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  const int B = emissions.dims(2);
  const int mS = noisytarget.dims(0);
  const int mkS = knoisytarget.dims(0);

  int top_k = top_k_ <= 0 ? N : std::min(std::max(2, top_k_), N);

  //fl::Variable scaling;
  //if (noiselm_.autoScale() == true) {
  //  scaling = fl::log(fl::sum(fl::exp(emissions),{0}));
  //} else {
  //  scaling = fl::Variable((af::constant(noiselm_.scaleValue(),1,T,B,1)),false);
  //}

  //std::vector<float> scaling_v(scaling.elements());
  std::vector<float> emissions_v(emissions.elements());
  std::vector<float> transitions_v(transitions.elements());
  std::vector<int> noisytarget_v(noisytarget.elements());
  std::vector<int> knoisytarget_v(knoisytarget.elements());
  emissions.host(emissions_v.data());
  transitions.host(transitions_v.data());
  noisytarget.host(noisytarget_v.data());
  knoisytarget.host(knoisytarget_v.data());
  //scaling.host(scaling_v.data());

  //std::vector<ForceAlignBeamNoiseNodeStats> stats_node = {ForceAlignBeamNoiseNodeStats()};
  //std::vector<ForceAlignBeamNoiseNode> true_node = stats_node;
  //ForceAlignBeamNoiseBatchData<ForceAlignBeamNoiseNode> data = ForceAlignBeamNoiseBatchData<ForceAlignBeamNoiseNodeStats>();
  //pass thre correct class in the template
  //if (compute_stats) {
  // data = std::make_shared<ForceAlignBeamNoiseBatchData<ForceAlignBeamNoiseNodeStats>>();
  //} else{
  //  data = std::make_shared<ForceAlignBeamNoiseBatchData<ForceAlignBeamNoiseNode>>();
  //}

  auto data = std::make_shared<ForceAlignBeamNoiseStatsBatchData>();
  data->batch.resize(B);

#pragma omp parallel for num_threads(B)
  for(int b = 0; b < B; b++) {
    auto emissions_p = emissions_v.data() + b*N*T;
    auto noisytarget_p = noisytarget_v.data() + b*mS;
    auto knoisytarget_p = knoisytarget_v.data() + b*mkS;
    //auto scaling_p = scaling_v.data() + b*T;
    const int S = w2l::getTargetSize(noisytarget_p, mS);
    const int kS = w2l::getTargetSize(knoisytarget_p, mkS);

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    // +root +end
    hyps.resize(T+2);
    std::vector<double> maxscores(T);


    NoiseTrieNode* keytrieroot = lex_->root(); // start with a silence root
    hyps.at(0).emplace_back(-1, 0,  -1,  -1, nullptr, keytrieroot->child(0), nullptr, -1, N-1, kS, 0, T); // We remove rep token
    //hyps.at(0).push_back({-1, 0,  -1,  -1, nullptr, keytrieroot->child(0), nullptr, -1, 0,0,0});

    //std::cout << "sould be 0: " << *hyps.at(0).back().nb_token_ptr << std::endl;
    double noisescore;
    double base_noisescore;
    double score;
    double baseScore;
    long noiselmstate;
    int nb_remaining_target;
    int nb_remaining_frame;
    NoiseTrieLabel *key = nullptr; //useless ?
    NoiseTrieNode *letter = nullptr;

    std::vector<ForceAlignBeamNoiseNodeStats> newhyps;


    std::vector<int> idx_unsort(N), idx_sorted(N);
    std::iota(idx_unsort.begin(), idx_unsort.end(), 0);
    idx_sorted = idx_unsort;
    int K;

    double cumul_maxscore = 0.0;

    int idx_nhyp;
    newhyps.reserve(B_ * N * 3);
    for(long t = 0; t < T; t++) {
      newhyps.clear();
      //newhyps.resize(B_ * N * 3);

      if (top_k < N){ // prune top k
        idx_sorted = idx_unsort;
        std::partial_sort( idx_sorted.begin(), idx_sorted.begin() + top_k, idx_sorted.end(), Comp(emissions_p + t*N) );
      }
      
      idx_nhyp = 0;
      //std::cout << "T " << t << std::endl;
      for(ForceAlignBeamNoiseNodeStats& prev : hyps.at(t)) {
        K = top_k;
        if (top_k < N && prev.l >= 0 && (std::find(idx_sorted.begin(), idx_sorted.begin() + top_k, prev.l) == idx_sorted.begin() + top_k)){
          idx_sorted[K] = prev.l;
          K++;
        }

        //for(long i = 0; i < N; i++) {
        for( int idx = 0; idx < K; idx++ ) {
          int& i = idx_sorted[idx];
          //std::cout << "i " << i << std::endl;
          baseScore = prev.score + emissions_p[t*N+i] + (prev.l >= 0 ? transitions_v[i*N+prev.l] : 0.);

          if (i == 28){ // rep1 label
          //if (i == 2){ // rep1 label
            letter = prev.letter;
          }
          else{
            letter = keytrieroot->child(i); // works for letter level
          }

          if (prev.l == -1){ // generate the first hypotheses except the rep label
            if (i != 28){
            //if (i != 2){  
              if (noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], letter->idx()) != -std::numeric_limits<double>::infinity() // If the next letter can be swapped
                  || noiselm_.scoreDeletion(letter->idx()) != -std::numeric_limits<double>::infinity() //if the next letter can be deleted
                  || (noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t + 1]) && (prev.knoisytarget_t + 2 < kS) // if the next letter can be inserted
                        && noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 2], letter->idx()) != -std::numeric_limits<double>::infinity() // and then the next one can be swapped or deleted (useless to verify since already done)
                       )
                  )
              {
                //newhyps.push_back({i, baseScore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, prev.nb_subs, prev.nb_ins, prev.nb_del});
                //std::cout << "prev -1" << std::endl;
                //ForceAlignBeamNoiseNodeStats& newnode = newhyps[idx_nhyp];
                //newnode.set(i, baseScore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, t+1);
                //idx_nhyp++;
                newhyps.emplace_back(i, baseScore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, t+1);
              }
            }
          } else{
            nb_remaining_target = kS - (prev.noisytarget_t + 1);
            nb_remaining_frame = T - (t + 1);
            if (prev.l == i) {
              // we force to follow the target if we don't have enough frames to finish it. Except for 1 letter if we allow insertion.
              if (nb_remaining_frame >= nb_remaining_target){ //THE PROBLEM IS HERE
                if (nb_remaining_frame - 1 >= nb_remaining_target
                  || (noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t], letter->idx()) != -std::numeric_limits<double>::infinity()) // If the next letter can be swapped
                  || (noiselm_.scoreDeletion(letter->idx()) != -std::numeric_limits<double>::infinity()
                      && (nb_remaining_frame - 1 >= nb_remaining_target)
                      ) //if the next letter can be deleted
                  || (noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]) != -std::numeric_limits<double>::infinity() && (prev.knoisytarget_t + 2 < kS) // if the next letter can be inserted
                        && noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], letter->idx()) != -std::numeric_limits<double>::infinity() // and then the next one can be swapped or deleted (useless to verify since already done)
                     )
                  )
                {
                  //newhyps.push_back({i, baseScore, prev.noisytarget_t, prev.knoisytarget_t, &prev, prev.letter, key, noiselmstate, prev.nb_subs, prev.nb_ins, prev.nb_del});
                  //std::cout << "same" << std::endl;
                  //ForceAlignBeamNoiseNodeStats& newnode = newhyps[idx_nhyp];
                  //newnode.set(i, baseScore, prev.noisytarget_t, prev.knoisytarget_t, &prev, prev.letter, key, noiselmstate, t+1);
                  //idx_nhyp++;
                  newhyps.emplace_back(i, baseScore, prev.noisytarget_t, prev.knoisytarget_t, &prev, prev.letter, key, noiselmstate, t+1);
                }
              }
            } else {
              // or we change the letter, hence the key
              if (noiselm_.allowInsertion() && prev.knoisytarget_t + 2 < kS){ //a letter has been added to the noisy transcription. +1 only because we can 
                if (noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 2], letter->idx()) != -std::numeric_limits<double>::infinity() // If the next letter can be swapped
                  || (noiselm_.scoreDeletion(letter->idx()) != -std::numeric_limits<double>::infinity()
                      && (nb_remaining_frame - 1 >= nb_remaining_target - 2) 
                      ) //if the next letter can be deleted
                  || (noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t + 2]) != -std::numeric_limits<double>::infinity() && (prev.knoisytarget_t + 4 < kS) // if the next letter can be inserted
                        && noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 3], letter->idx()) != -std::numeric_limits<double>::infinity() // and then the next one can be swapped or deleted (useless to verify since already done)
                     )
                  )
                {
                  base_noisescore = noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]);
                  noisescore = noiselm_.allowSwap() ? base_noisescore + noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], prev.letter->idx()) : base_noisescore;
                  //newhyps.push_back({i, baseScore + noisescore, prev.noisytarget_t + 2, prev.knoisytarget_t + 2, &prev, letter, key, noiselmstate, prev.nb_subs + 1, prev.nb_ins + 1, prev.nb_del});
                  if (noisescore != -std::numeric_limits<double>::infinity()){
                    //ForceAlignBeamNoiseNodeStats& newnode = newhyps[idx_nhyp];
                    //newnode.set(i, baseScore + noisescore, prev.noisytarget_t + 2, prev.knoisytarget_t + 2, &prev, letter, key, noiselmstate, t+1);
                    //idx_nhyp++;
                    newhyps.emplace_back(i, baseScore + noisescore, prev.noisytarget_t + 2, prev.knoisytarget_t + 2, &prev, letter, key, noiselmstate, t+1);
                    newhyps.back().updateSub(knoisytarget_p[prev.knoisytarget_t + 1], prev.letter->idx());
                    newhyps.back().updateIns(knoisytarget_p[prev.knoisytarget_t]);
                  }
                }
              }

              base_noisescore = noiselm_.allowInsertion() ? noiselm_.scoreNoInsertion() : 0.; //no letter is inserted
              //base_noisescore = 0.0;
              if (noiselm_.allowDeletion() && prev.knoisytarget_t < kS){ // for now allow only 1 deletion and one last deletion
                // Verify that the proposed hyp is probable
                if (noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t], letter->idx()) != -std::numeric_limits<double>::infinity() // If the next letter can be swapped
                    || (noiselm_.scoreDeletion(letter->idx()) != -std::numeric_limits<double>::infinity()
                        && (nb_remaining_frame - 1 >= nb_remaining_target)
                        ) //if the next letter can be deleted
                    || (noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]) != -std::numeric_limits<double>::infinity() && (prev.knoisytarget_t + 2 < kS) // if the next letter can be inserted
                        && noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], letter->idx()) != -std::numeric_limits<double>::infinity() // and then the next one can be swapped or deleted (useless to verify since already done)
                       )
                   ) 
                {
                  //we allow a deletion only if we have enough frames to finish the sentence.
                  if (nb_remaining_frame >= nb_remaining_target){
                    noisescore = base_noisescore + noiselm_.scoreDeletion(prev.letter->idx());           
                    //newhyps.push_back({i, baseScore + noisescore, prev.noisytarget_t, prev.knoisytarget_t, &prev, letter, key, noiselmstate, prev.nb_subs, prev.nb_ins, prev.nb_del + 1});
                    if (noisescore != -std::numeric_limits<double>::infinity()){
                      //ForceAlignBeamNoiseNodeStats& newnode = newhyps[idx_nhyp];
                      //newnode.set(i, baseScore + noisescore, prev.noisytarget_t, prev.knoisytarget_t, &prev, letter, key, noiselmstate, t+1);
                      //idx_nhyp++;
                      newhyps.emplace_back(i, baseScore + noisescore, prev.noisytarget_t, prev.knoisytarget_t, &prev, letter, key, noiselmstate, t+1);
                      newhyps.back().updateDel(prev.letter->idx());
                    }
                  }
                }
              }

              if (noiselm_.allowSwap() && prev.knoisytarget_t + 1 < kS){
                // Verify that the proposed hyp is probable. Otherwise most of the beam hyps are useless if the noise model is sparse.
                if (noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], letter->idx()) != -std::numeric_limits<double>::infinity() // If the next letter can be swapped
                    || (noiselm_.scoreDeletion(letter->idx()) != -std::numeric_limits<double>::infinity()
                        && (nb_remaining_frame - 1 >= nb_remaining_target - 1) 
                        ) //if the next letter can be deleted
                    || (noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t + 1]) != -std::numeric_limits<double>::infinity() && (prev.knoisytarget_t + 3 < kS) // if the next letter can be inserted
                        && noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 2], letter->idx()) != -std::numeric_limits<double>::infinity() // and then the next one can be swapped or deleted (useless to verify since already done)
                       )
                    )
                {
                  noisescore = base_noisescore + noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t], prev.letter->idx());
                  //newhyps.push_back({i, baseScore + noisescore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, prev.nb_subs + 1, prev.nb_ins, prev.nb_del});
                  //std::cout << "swap" << std::endl;
                  if (noisescore != -std::numeric_limits<double>::infinity()){
                    //ForceAlignBeamNoiseNodeStats& newnode = newhyps[idx_nhyp];
                    //newnode.set(i, baseScore + noisescore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, t+1);
                    //idx_nhyp++;
                    newhyps.emplace_back(i, baseScore + noisescore, prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev, letter, key, noiselmstate, t+1);
                    newhyps.back().updateSub(knoisytarget_p[prev.knoisytarget_t], prev.letter->idx());
                  }
                }
              }  
            }
          }
        }
      }
      //newhyps.resize(idx_nhyp);


      // offset scores
      double maxscore = -std::numeric_limits<double>::infinity();
      for(ForceAlignBeamNoiseNodeStats& hyp : newhyps) {
        maxscore = std::max(maxscore, hyp.score);
      }
      cumul_maxscore += maxscore;
      for(ForceAlignBeamNoiseNodeStats& hyp : newhyps) {
        hyp.score -= maxscore;
        hyp.maxscore = maxscore; //useless now ? pass pointeur ?
        hyp.cumul_maxscore = cumul_maxscore;
      }
      maxscores[t] = maxscore;

      // prune hyps
      if(threshold_ > 0) {
        float npruned = 0;
        for(size_t i = 0; i < newhyps.size(); i++) {
          if(newhyps.at(i).score > maxscore - threshold_) {
            if(i != npruned) {
              newhyps.at(npruned) = newhyps.at(i);
            }
            npruned++;
          }
        }
        newhyps.resize(npruned);
      }

      // merge identical nodes
      std::sort(newhyps.begin(), newhyps.end(), [](ForceAlignBeamNoiseNodeStats& a, ForceAlignBeamNoiseNodeStats& b) {
          if(a.l == b.l) { // same as a.lex == b.lex but count the rep1 token
            if(a.knoisytarget_t == b.knoisytarget_t) {
              return a.score > b.score;
            } else {
              return a.noisytarget_t < b.noisytarget_t;
            }
          } else {
            return a.l < b.l;
          }
        });

      long headidx = 0;
      long nhyp = newhyps.size();
      //std::cout << "nhyp before merge " << nhyp << std::endl;

      double coeff;
      double head_score_ini = newhyps.at(headidx).score;
      for(long h = 1; h < nhyp; h++) {
        ForceAlignBeamNoiseNodeStats& elem = newhyps.at(h);
        ForceAlignBeamNoiseNodeStats& head = newhyps.at(headidx);

        if( (head.l == elem.l) && (head.knoisytarget_t == elem.knoisytarget_t)) { //maybe to change when key level
          if(head.merge_a < 0) {
            head.merge_a = merged.size();
            merged.push_back(head.parent); /* note: parent is in hyps */
            merged_score.push_back(head.score);
          }
          head.merge_b = merged.size();
          merged.push_back(elem.parent); /* note: parent is in hyps */
          merged_score.push_back(elem.score);
          //std::cout << "head.score " << head.score << std::endl;
          //std::cout << "elem.score " << elem.score << std::endl;
          head.score = logadd(head.score, elem.score);
          //std::cout << "head.score " << head.score << std::endl;
          coeff = exp(elem.score - head_score_ini);
          if (std::isnan(coeff)){
            std::cout << "NAN at t=" << t << std::endl;
            std::cout << "elem score " << elem.score << " head score " << head_score_ini << std::endl;
          }
          head.mergeWStats(elem, coeff);
          //head.nb_subs += elem.nb_subs * coeff;
          //head.nb_ins += elem.nb_ins * coeff;
          //head.nb_del += elem.nb_del * coeff;
          //sum_coeff += coeff;
        } else {
          head.finalizeWStats();
          headidx++;
          if(headidx != h) {
            newhyps.at(headidx) = newhyps.at(h);
          }
          //sum_coeff = 1.0;
          head_score_ini = newhyps.at(headidx).score;
        }
        if (h == nhyp - 1){ // finalize the score for the last considered head
          head.finalizeWStats();
        }
      }
      nhyp = headidx+1;
      //std::cout << "nhyp after " << nhyp << std::endl;
      // beam it
      std::sort(newhyps.begin(), newhyps.begin()+nhyp, [](ForceAlignBeamNoiseNodeStats& a, ForceAlignBeamNoiseNodeStats& b) { return a.score > b.score; });
      nhyp = std::min(nhyp, B_);
      hyps.at(t+1).insert(hyps.at(t+1).end(), newhyps.begin(), newhyps.begin() + nhyp);
      //for(ForceAlignBeamNoiseNodeStats& hyp : hyps.at(t+1)) {
      //    std::cout << "l " << hyp.l << " s " << hyp.score << " M " << hyp.maxscore << std::endl;
      //}
    }
    //for(ForceAlignBeamNoiseNode& hyp : hyps.at(T)) {
    //    std::cout << "s " << hyp.score << std::endl;
    //    std::cout << "M " << hyp.maxscore << std::endl;
    //}
    // finish
    //-1e220 -> we assign a probability even if the beam failed. To change, maybe it causes a bug
    //ForceAlignBeamNoiseNode fini(-1, -1e200, -1, -1, nullptr, nullptr, nullptr, -1);
    ForceAlignBeamNoiseNodeStats fini("fini");
    for(ForceAlignBeamNoiseNodeStats &prev : hyps.at(T)) {
      if(fini.merge_a < 0) {
        fini.merge_a = merged.size();
        fini.merge_b = merged.size() - 1;
      }

      double score;
      double noisescore;
      double base_noisescore=0.;
      //onlyHypWithTargetFinished is deprecated
      bool merge_it = false;
      //auto nb_subs = prev.nb_subs;
      //auto nb_ins = prev.nb_ins;
      //auto nb_del = prev.nb_del;
      if (prev.knoisytarget_t + 1 == kS && noiselm_.allowSwap()){ // no insertion + swap current + no insertion after
        base_noisescore = noiselm_.allowInsertion() ? 2.0 * noiselm_.scoreNoInsertion() : 0.; //no letter is inserted
        //base_noisescore = 0.0;
        noisescore = base_noisescore + noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t], prev.letter->idx());
        prev.updateSub(knoisytarget_p[prev.knoisytarget_t], prev.letter->idx());
        merge_it = true;
      } else if (prev.knoisytarget_t + 2 == kS && noiselm_.allowInsertion()){ // insertion + swap current + no insertion
        base_noisescore = noiselm_.scoreNoInsertion() + noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]); //no letter is inserted
        //base_noisescore = noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]);
        noisescore = noiselm_.allowSwap() ? base_noisescore + noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t + 1], prev.letter->idx()) : base_noisescore;
        prev.updateIns(knoisytarget_p[prev.knoisytarget_t]);
        prev.updateSub(knoisytarget_p[prev.knoisytarget_t + 1], prev.letter->idx());
        merge_it = true;
      } else if (prev.knoisytarget_t == kS && noiselm_.allowDeletion()){ //deletion of the previous letter + no insertion after
        base_noisescore = noiselm_.allowInsertion() ? noiselm_.scoreNoInsertion() : 0.;
        //base_noisescore = 0.0;
        noisescore = base_noisescore + noiselm_.scoreDeletion(prev.letter->idx());
        prev.updateDel(prev.letter->idx());
        merge_it = true;
      }
      if (merge_it) {
        score = prev.score + noisescore;
        fini.merge_b = merged.size();
        merged.push_back(&prev);
        merged_score.push_back(score);
        //std::cout << fini.score << " " << score << std::endl;
        fini.score = logadd(fini.score, score);
        //std::cout << fini.score << std::endl;

        int n = fini.merge_b - fini.merge_a + 1;
        //fini.nb_subs = ((n-1) * fini.nb_subs + nb_subs) / n;
        //fini.nb_ins = ((n-1) * fini.nb_ins + nb_ins) / n;
        //fini.nb_del = ((n-1) * fini.nb_del + nb_del) / n;
      }
    }

    //std::cout << "FIN " << fini.score << std::endl;

    double sum_of_maxscores = 0.0;
    //std::cout << "MAXSCORES " << std::endl;
    int k=0;
    for(double& score : maxscores) {
      sum_of_maxscores += score;
      //std::cout << k << " " << score <<std::endl;
      //k++;
    }
    //std::cout << "MAX ";
    //std::cout << sum_of_maxscores << std::endl;

    if (sum_of_maxscores != -std::numeric_limits<double>::infinity()){
      fini.score += sum_of_maxscores;
    }
    hyps.at(T+1).push_back(std::move(fini));
    //std::cout << "FIN 2 " << fini.score << std::endl;
  }

  std::vector<float> result_p(B);
  for(int b = 0; b < B; b++) {
    result_p[b] = data->batch[b].hyps.at(T+1).at(0).score;
  }
  auto result = af::array(B, result_p.data());
  auto grad_func = [this,data](
    std::vector<fl::Variable>& inputs,
    const fl::Variable& goutput) {
    this->backward(inputs, goutput, data, false);
  };

  std::vector<fl::Variable> inputs = {emissions, transitions, noisytarget, knoisytarget};
  const std::vector<fl::Variable> noiselmParams = noiselm_.params();
  inputs.insert(inputs.end(), noiselmParams.begin(), noiselmParams.end());

  return fl::Variable(result, inputs, grad_func, std::make_shared<ForceAlignBeamNoiseStatsVariablePayload>(data));
}

static void dlogadd(std::vector<double>& score, std::vector<double>& gscore, double g)
{
  double m = -std::numeric_limits<double>::infinity();
  for(size_t i = 0; i < score.size(); i++) {
    m = std::max(m, score[i]);
  }
  double sum = 0;
  for(size_t i = 0; i < score.size(); i++) {
    sum += exp(score[i]-m);
  }
  for(size_t i = 0; i < score.size(); i++) {
    if (m == -std::numeric_limits<double>::infinity()){ //when no hypothesis have been found. actually should no occurs...
      gscore[i] == 0.0;
    } else{
      gscore[i] = exp(score[i]-m)/sum*g;
    } 
  }
}


static void accnode(ForceAlignBeamNoiseNodeStats& node, ForceAlignBeamNoiseNodeStats& prev, float* gemissions_p, float* gtransitions_p, long t, long T, long N, NoiseLMLetterSwapUnit& noiselm, double g)
{
  if(t >= 0 && t < T) {
    gemissions_p[t*N+node.l] += g;
    if(prev.l >= 0) {
      gtransitions_p[node.l*N+prev.l] += g;
    }
    //to change later!
    //if(node.key) {
    //  noiselm.accGradScore(node.lm, g);
    //}
  }
  prev.gscore += g;
  prev.active = true;
}

static void print_node(std::ostream& f, long t, ForceAlignBeamNoiseNodeStats& node)
{
  if(t >= 0) {
    f << "node_" << t << "_" << (long)(&node);
  } else {
    f << "node_R" << "_" << (long)(&node);
  }
}

static void define_node(std::ostream& f, long t, ForceAlignBeamNoiseNodeStats& node)
{
  print_node(f, t, node);
  f << " [label_l=\"" << node.l << "\"  label_t=\"" << node.knoisytarget_t << "\"];" << std::endl;
}

static void connect_nodes(std::ostream& f, long t1, ForceAlignBeamNoiseNodeStats& node1, long t2, ForceAlignBeamNoiseNodeStats& node2)
{
  print_node(f, t1, node1);
  f << " -> ";
  print_node(f, t2, node2);
  f << ";" << std::endl;
}

//void ForceAlignBeamNoise::showEditOps(const fl::Variable& output, int64_t b)
//{
//  auto payload = output.getPayload();
//  if(!payload) {
//    throw std::invalid_argument("expecting a payload on provided Variable");
//  }
//  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;
//  auto& hyps = data->batch[b].hyps;
//  fini = hyps.at(T+1).at(0);
//
//  std::cout << 
//}

void ForceAlignBeamNoiseStats::showTarget(const fl::Variable& output, int64_t b, std::ostream& f)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;
  //auto& merged_noisytarget_score = data->batch[b].merged_noisytarget_score;

  f << "digraph decfal {" << std::endl;
  long T = hyps.size()-2;
  for(std::vector<ForceAlignBeamNoiseNodeStats>& hyps_t : hyps) {
    for(ForceAlignBeamNoiseNodeStats& node : hyps_t) {
      node.gscore = 0;
      node.active = false;
    }
  }
  hyps.at(T+1).at(0).active = true;
  define_node(f, T, hyps.at(T+1).at(0));

  std::vector<double> sub_merged_score;
  for(long t = T; t >= 0; t--) {
    for(ForceAlignBeamNoiseNodeStats& node : hyps.at(t+1)) {
      if(node.active) {
        if(node.merge_a >= 0) {
          long n = node.merge_b-node.merge_a+1;
          sub_merged_score.resize(n);
          std::copy(merged_score.begin()+node.merge_a, merged_score.begin()+node.merge_b+1, sub_merged_score.begin());
          for(long idx = 0; idx < n; idx++) {
            if(sub_merged_score[idx] != -std::numeric_limits<double>::infinity()) {
              if(!merged.at(node.merge_a+idx)->active) {
                merged.at(node.merge_a+idx)->active = true;
                define_node(f, t-1, *merged.at(node.merge_a+idx));
              }
              f << "subscore: " << sub_merged_score[idx] << " ";
              connect_nodes(f, t-1, *merged.at(node.merge_a+idx), t, node);
            }
          }
        } else {
          if(!node.parent->active) {
            node.parent->active = true;
            define_node(f, t-1, *node.parent);
          }
          f << "score: " << node.score << " ";
          connect_nodes(f, t-1, *node.parent, t, node);
        }
      }
    }
  }
  f << "}" << std::endl;
}




/*
struct pathsInfoBeamForward {
  typedef std::tuple<std::vector<int>, double, std::vector<ForceAlignBeamNoiseNodeStats*>, int > pathTuple;
  typedef std::tuple<std::vector<int>, double> simplePathTuple;
  std::vector<pathTuple> pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing the adresses of the nodes that have lead to the path.
  //             We can have multiple nodes because of the merging operation during the forward pass and because of the multiple alignements.
  //  - get<2> : the current target of the nodes in the vector
  

  pathsInfoBeamForward() {};
  //pathsInfoBeam(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
    // Add the path if not present. Otherwise add the score the already stored path.
    auto it_existing_path = std::find_if(pathsInfo.begin(), pathsInfo.end(),
                          [&path, &node_ptr](const pathTuple& tuple) {
                            return (std::get<0>(tuple) == path);
                              //return ((std::get<0>(tuple) == path) && (std::get<3>(tuple) == node_ptr->knoisytarget_t));
                          });
    // Verify if the path is present by comparing the first elemnents of the tuples.
    
    if (it_existing_path == pathsInfo.end()) { //if the path is not present, we add it
      std::vector<ForceAlignBeamNoiseNodeStats*> single_vect{node_ptr};
      //pathTuple new_tuple = std::make_tuple(path, value, single_vect);
      pathTuple new_tuple = std::make_tuple(path, value, single_vect, node_ptr->knoisytarget_t);
      pathsInfo.push_back(new_tuple);
    } else{
      std::get<1>(*it_existing_path) = logadd(std::get<1>(*it_existing_path), value); // Logadd the scores
      std::get<2>(*it_existing_path).push_back(node_ptr); // Keep track of the node that has generated the path.

      //std::get<3>(*it_existing_path) = logadd(std::get<3>(*it_existing_path), node_ptr->score);
    }
  }

  void sortIt(){
    std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
        return std::get<1>(a) > std::get<1>(b);
      });
  }

  void beamIt(int beam_size){
    sortIt();
    if (beam_size < pathsInfo.size()){ // if the vector is too big.
      // we sort it
      pathsInfo.resize(beam_size);
    }
  }

  std::map<ForceAlignBeamNoiseNodeStats*, std::set<simplePathTuple> > getPathsByNode(){
  // Rearange pathsInfo by node. i.e. return a map where the keys are the nodes and the value is a vector of simplePathTuple.
    std::map<ForceAlignBeamNoiseNodeStats*, std::set<simplePathTuple> > MapByNode;
    for (auto const& path_tuple : pathsInfo){
      for (auto const& node : std::get<2>(path_tuple)){
        simplePathTuple new_tuple = std::make_tuple(std::get<0>(path_tuple), std::get<1>(path_tuple));
        auto it_existing = MapByNode.find(node);
        if (it_existing == MapByNode.end()) { //if node is not present, we add it.
          MapByNode[node] = {new_tuple};
        } else{
          (it_existing->second).insert(new_tuple);
        }
      }
    }
    return MapByNode;
  }


  std::vector<simplePathTuple> getResult(){ // remove node ptr and reverse paths
    std::vector< simplePathTuple> result;
    for (auto const& path_tuple : pathsInfo){
      result.push_back(std::make_tuple(std::get<0>(path_tuple),std::get<1>(path_tuple)));
    }
    return result;
  }

};

*/

struct pathsInfoBeamForward {
  typedef std::tuple<std::vector<int>, double, std::vector<std::tuple<ForceAlignBeamNoiseNodeStats*, double>>, double> pathTuple;
  typedef std::tuple<std::vector<int>, double> simplePathTuple;
  std::vector<pathTuple> pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing the adresses of the nodes that have lead to the path.
  //             We can have multiple nodes because of the merging operation during the forward pass and because of the multiple alignements.
  //  - get<2> : the current target of the nodes in the vector
  

  pathsInfoBeamForward() {};
  //pathsInfoBeam(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
    // Add the path if not present. Otherwise add the score the already stored path.
    auto it_existing_path = std::find_if(pathsInfo.begin(), pathsInfo.end(),
                          [&path](const pathTuple& tuple) {
                            return (std::get<0>(tuple) == path);
                          });
    
    if (it_existing_path == pathsInfo.end()) { //if the path is not present, we add it
      std::vector<std::tuple<ForceAlignBeamNoiseNodeStats*, double>> single_vect{std::make_tuple(node_ptr, value)};
      //pathTuple new_tuple = std::make_tuple(path, value, single_vect, node_ptr->score + node_ptr->maxscore);
      //pathTuple new_tuple = std::make_tuple(path, value, single_vect, (node_ptr->score + node_ptr->maxscore) * node_ptr->gscore);
      pathTuple new_tuple = std::make_tuple(path, value, single_vect , value + node_ptr->gscore - (node_ptr->score + node_ptr->cumul_maxscore));
      pathsInfo.push_back(new_tuple);
    } else{
      std::get<1>(*it_existing_path) = logadd(std::get<1>(*it_existing_path), value); // Logadd the total score of the path


      //we search if this node has already contributed to this path.
      
      auto it_exist_node = std::find_if(std::get<2>(*it_existing_path).begin(), std::get<2>(*it_existing_path).end(),
                          [&node_ptr](const std::tuple<ForceAlignBeamNoiseNodeStats*, double>& current_node_value) {
                            return (std::get<0>(current_node_value) == node_ptr);
                          });

      if (it_exist_node == std::get<2>(*it_existing_path).end()) { // if the node has never contributed to this path
        std::get<2>(*it_existing_path).push_back(std::make_tuple(node_ptr, value)); //we add a new reference to this node with the value
        //std::get<3>(*it_existing_path) = logadd(std::get<3>(*it_existing_path), node_ptr->score + node_ptr->maxscore);
        //std::get<3>(*it_existing_path) += (node_ptr->score + node_ptr->maxscore) * node_ptr->gscore;
        std::get<3>(*it_existing_path) = logadd(std::get<3>(*it_existing_path), value + node_ptr->gscore - (node_ptr->score + node_ptr->cumul_maxscore));
      } else{
        std::get<1>(*it_exist_node) = logadd(std::get<1>(*it_exist_node), value); // else we logadd the contribution of the node to this path.
      }

    }

  }

  void sortIt(bool with_node_score){
    if (with_node_score){
        std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
        //return std::get<1>(a) > std::get<1>(b);
          //return std::get<1>(a) + std::get<3>(a) > std::get<1>(b) + std::get<3>(b);
        double contrib1 = -std::numeric_limits<double>::infinity();
        double contrib2 = -std::numeric_limits<double>::infinity();

        for (auto& node_ptr_value : std::get<2>(a)){
          auto& node_ptr = std::get<0>(node_ptr_value);
          auto& value = std::get<1>(node_ptr_value);
          contrib1 = logadd(contrib1, value + node_ptr->gscore - (node_ptr->score + node_ptr->cumul_maxscore));
        }

        for (auto& node_ptr_value : std::get<2>(b)){
          auto& node_ptr = std::get<0>(node_ptr_value);
          auto& value = std::get<1>(node_ptr_value);
          contrib2 = logadd(contrib2, value + node_ptr->gscore - (node_ptr->score + node_ptr->cumul_maxscore));
        }


        return contrib1 > contrib2;
      });
    } else{
          std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
          return std::get<1>(a) > std::get<1>(b);
      });
    }
  }

  void beamIt(int beam_size, bool with_node_score){
    sortIt(with_node_score);
    if (beam_size < pathsInfo.size()){ // if the vector is too big.
      // we sort it
      pathsInfo.resize(beam_size);
    }
    std::cout << "--------------------------------------------" << std::endl;
    for (auto& path:pathsInfo){
      std::cout << std::get<1>(path) << " " << std::get<3>(path) << "[ ";
      for (auto& tuple : std::get<2>(path)){
        double score = std::get<0>(tuple)->score + std::get<0>(tuple)->cumul_maxscore;
        double next_pot = std::get<0>(tuple)->gscore - score;
        std::cout << "(" << std::get<0>(tuple) << ", " << std::get<0>(tuple)->gscore << ", " << score << ", " << next_pot << std::get<1>(tuple) << "),";
      }
      std::cout << "]" << std::endl;
    }
  }

  std::map<ForceAlignBeamNoiseNodeStats*, std::set<simplePathTuple> > getPathsByNode(){
  // Rearange pathsInfo by node. i.e. return a map where the keys are the nodes and the value is a vector of simplePathTuple.
    std::map<ForceAlignBeamNoiseNodeStats*, std::set<simplePathTuple> > MapByNode;
    for (auto const& path_tuple : pathsInfo){
      for (auto const& node_and_value : std::get<2>(path_tuple)){
        auto& node = std::get<0>(node_and_value);
        auto& value = std::get<1>(node_and_value);
        simplePathTuple new_tuple = std::make_tuple(std::get<0>(path_tuple), value);
        auto it_existing = MapByNode.find(node);
        if (it_existing == MapByNode.end()) { //if node is not present, we add it.
          MapByNode[node] = {new_tuple};
        } else{
          (it_existing->second).insert(new_tuple);
        }
      }
    }
    return MapByNode;
  }


  std::vector<simplePathTuple> getResult(bool reverse){ // remove node ptr and reverse paths
    std::vector< simplePathTuple> result;
    for (auto const& path_tuple : pathsInfo){
      auto path = std::get<0>(path_tuple);
      if (reverse){
        std::reverse(path.begin(), path.end());
      }
      result.push_back(std::make_tuple(path,std::get<1>(path_tuple)));
    }
    return result;
  }
};


std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsForward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  assert(hyps.size() >= 3);
  const size_t T = hyps.size()-2;
  for(auto & hyps_t : hyps) {
    for(auto & node : hyps_t) {
      node.active = false;
      node.gscore = 0.;
    }
  }

  auto& fini = hyps.at(T+1).at(0);
  fini.active = true;
  fini.gscore = fini.score;

  std::vector<double> sub_merged_gscore;
  std::vector<double> sub_merged_score;
  for (int t = T+1; t>=1; t--){
    auto& hyps_t = hyps.at(t);
    for (auto &node : hyps_t){
      if (node.active == true){
        if (node.merge_a >= 0){
          int n = node.merge_b - node.merge_a + 1;
          sub_merged_gscore.resize(n);
          sub_merged_score.resize(n);
          std::copy(merged_score.begin() + node.merge_a, merged_score.begin() + node.merge_b+1, sub_merged_score.begin());
          dlogadd(sub_merged_score, sub_merged_gscore, node.gscore);
          for(long idx = 0; idx < n; idx++) {
            if(sub_merged_score[idx] != -std::numeric_limits<double>::infinity()) {
              auto& next_node = merged[idx + node.merge_a];
              next_node->active = true;
              next_node->gscore += sub_merged_gscore[idx];
            }
          }
        } else{
          node.parent->gscore += node.gscore;
          node.parent->active = true;
        }
      }
    }
  }

  //for (int t = 1; t<=T+1; t++){
  //  auto& hyps_t = hyps.at(t);
  // double tot_grad = 0.0;
  //  std::cout << "--------- " << t << " ---------" << std::endl;
  //  for (auto &node : hyps_t){
  //    tot_grad += node.gscore;
  //    std::cout << node.gscore << std::endl;
  //  }
  //  std::cout << "TOTAL: " << tot_grad << std::endl;
  //}



  pathsInfoBeamForward pathsInfo_current;
  pathsInfoBeamForward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamForward();
  std::vector<int> void_path(0); // a void path for the node 0.
  pathsInfo_current.addPathValue(void_path, 0.,  &hyps.at(0).at(0));
  for (int t = 1; t<=T+1; t++){
    std::cout << "------------- " <<  t <<  " ------------" << std::endl;
    pathsInfo_previous = pathsInfo_current;
    pathsInfo_current = pathsInfoBeamForward();
    auto paths_by_node = pathsInfo_previous.getPathsByNode();
    auto& hyps_t = hyps.at(t);
    for (auto& node : hyps_t){
      if (node.active == true){
        std::cout << &node << " " << node.gscore << std::endl;
        if (node.merge_a >= 0){
          int n = node.merge_b - node.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto& previous_node = merged[idx + node.merge_a];
            auto& demerged_score = merged_score[idx + node.merge_a];
            auto it_paths_tuple = paths_by_node.find(previous_node);
            if (it_paths_tuple != paths_by_node.end()){
              for (auto& path_tuple : it_paths_tuple->second){
                auto new_path = std::get<0>(path_tuple);
                if (new_path.size() == 0 || (new_path.back() != node.l && node.l != -1)){
                  new_path.push_back(node.l);
                }
                pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + node.maxscore 
                                                        + demerged_score - previous_node->score,  &node);
              }
            }
          }
        } else{
          auto& previous_node = node.parent;
          auto it_paths_tuple = paths_by_node.find(previous_node);
          if (it_paths_tuple != paths_by_node.end()){
            for (auto& path_tuple : it_paths_tuple->second){
              auto new_path = std::get<0>(path_tuple);
              if (new_path.size() == 0 || (new_path.back() != node.l && node.l != -1)){
                new_path.push_back(node.l);
              }
              pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + node.maxscore 
                                                        + node.score - previous_node->score,  &node);                              
            }
          }
        }
      }
    }
    pathsInfo_current.beamIt(beam_size, true);
  }
  return pathsInfo_current.getResult(false);
}

/*

std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsForward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  assert(hyps.size() >= 3);
  const size_t T = hyps.size()-2;
  for(auto & hyps_t : hyps) {
    for(auto & node : hyps_t) {
      node.active = false;
    }
  }

  auto& fini = hyps.at(T+1).at(0);
  fini.active = true;

  for (int t = T+1; t>=1; t--){
    auto& hyps_t = hyps.at(t);
    for (auto &node : hyps_t){
      if (node.active == true){
        if (node.merge_a >= 0){
          int n = node.merge_b - node.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto& next_node = merged[idx + node.merge_a];
            next_node->active = true;
          }
        } else{
          node.parent->active = true;
        }
      }
    }
  }

  pathsInfoBeamForward pathsInfo_current;
  pathsInfoBeamForward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamForward();
  std::vector<int> void_path(0); // a void path for the node 0.
  pathsInfo_current.addPathValue(void_path, 0.,  &hyps.at(0).at(0));
  for (int t = 1; t<=T+1; t++){
    //std::cout << t << "--------------> " << pathsInfo_previous.pathsInfo.size() << std::endl;
    pathsInfo_previous = pathsInfo_current;
    pathsInfo_current = pathsInfoBeamForward();
    auto paths_by_node = pathsInfo_previous.getPathsByNode();
    auto& hyps_t = hyps.at(t);
    for (auto& node : hyps_t){
      if (node.active == true){
        if (node.merge_a >= 0){
          int n = node.merge_b - node.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto& previous_node = merged[idx + node.merge_a];
            auto& demerged_score = merged_score[idx + node.merge_a];
            auto it_paths_tuple = paths_by_node.find(previous_node);
            if (it_paths_tuple != paths_by_node.end()){
              for (auto& path_tuple : it_paths_tuple->second){
                auto new_path = std::get<0>(path_tuple);
                if (new_path.size() == 0 || (new_path.back() != node.l && node.l != -1)){
                  new_path.push_back(node.l);
                }
                pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + node.maxscore 
                                                        + demerged_score - previous_node->score,  &node);
              }
            }
          }
        } else{
          auto& previous_node = node.parent;
          auto it_paths_tuple = paths_by_node.find(previous_node);
          if (it_paths_tuple != paths_by_node.end()){
            for (auto& path_tuple : it_paths_tuple->second){
              auto new_path = std::get<0>(path_tuple);
              if (new_path.size() == 0 || (new_path.back() != node.l && node.l != -1)){
                new_path.push_back(node.l);
              }
              pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + node.maxscore 
                                                        + node.score - previous_node->score,  &node);                              
            }
          }
        }
      }
    }
    pathsInfo_current.beamIt(beam_size, false);
  }
  return pathsInfo_current.getResult(false);
}

*/

///THE NEW ONE
///


struct pathsInfoBeamBackward {
  typedef std::tuple<std::vector<int>, double, std::map<ForceAlignBeamNoiseNodeStats*, double>, double> pathTuple;
  typedef std::tuple<std::vector<int>, double> simplePathTuple;
  std::vector<pathTuple> pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing the adresses of the nodes that have lead to the path.
  //             We can have multiple nodes because of the merging operation during the forward pass and because of the multiple alignements.
  

  pathsInfoBeamBackward() {};
  //pathsInfoBeam(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
    // Add the path if not present. Otherwise add the score of the already stored path.
    auto it_existing_path = std::find_if(pathsInfo.begin(), pathsInfo.end(),
                          [&path](const pathTuple& tuple) {return (std::get<0>(tuple) == path);});
    // Verify if the path is present by comparing the first elemnents of the tuples.
    
    if (it_existing_path == pathsInfo.end()) { //if the path is not present, we add it
      std::map<ForceAlignBeamNoiseNodeStats*, double> single_map = {{node_ptr, value}};
      pathTuple new_tuple = std::make_tuple(path, value, single_map, node_ptr->score);
      pathsInfo.push_back(new_tuple);
    } else{
      std::get<1>(*it_existing_path) = logadd(std::get<1>(*it_existing_path), value); // total score of the path currently

      auto& node_map = std::get<2>(*it_existing_path);
      auto it_exist_node = node_map.find(node_ptr);
      if (it_exist_node == node_map.end()){
        node_map[node_ptr] = value;
        //std::get<3>(*it_existing_path) = logadd(std::get<3>(*it_existing_path), node_ptr->score);
        //std::get<3>(*it_existing_path) = std::max(std::get<3>(*it_existing_path), node_ptr->score);
      } else{
        it_exist_node->second = logadd(it_exist_node->second, value);
      }
    }
  }


  void sortIt(){
    //we first compute the sorting criteria
    //
    
    for(auto& path_info : pathsInfo){
      double res = -std::numeric_limits<double>::infinity();
      for (auto& node_value : std::get<2>(path_info)){
        auto& node_ptr = node_value.first;
        auto& value = node_value.second;
        res = logadd(res, node_ptr->score + value);
        //res = std::max(res, node_ptr->score + value);
      }
      std::get<3>(path_info) = res;
    }
    

    std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
        //return std::get<1>(a) + std::get<2>(a)->score > std::get<1>(b) + std::get<2>(b)->score;
        //return std::get<1>(a) + std::get<3>(a) > std::get<1>(b) + std::get<3>(b);
        return std::get<3>(a) > std::get<3>(b);
      });
  }

  void beamIt(int beam_size){
    sortIt();
    if (beam_size < pathsInfo.size()){ // if the vector is too big.
      // we sort it
      pathsInfo.resize(beam_size);
    }
  }

  std::vector<simplePathTuple> getResult(){ // remove node ptr and reverse paths
    std::vector< simplePathTuple> result;
    for (auto const& path_tuple : pathsInfo){
      auto path = std::get<0>(path_tuple);
      std::reverse(path.begin(), path.end());
      result.push_back(std::make_tuple(path,std::get<1>(path_tuple)));
    }
    return result;
  }

};



std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data->batch[b];

  auto& hyps = data.hyps;
  auto& merged = data.merged;
  auto& merged_score = data.merged_score;


  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size()-2;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T+1).at(0));
  //std::cout << &hyps.at(T+1).at(0) << std::endl;
  double score_before_merge;
  double score_no_merge;
  long n;

  for(long t = T; t >= 0; t--) {
    //std::cout << t << "/" << T << std::endl;
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo){
      auto& prev_path = std::get<0>(path_info);
      //auto& prev_path_score = std::get<1>(path_info);
      auto& prev_nodes = std::get<2>(path_info);
      //just a new for here
      for (auto const& prev_node_info : prev_nodes){
        auto& prev_node = prev_node_info.first;
        auto& prev_path_score = prev_node_info.second;
        if(prev_node->merge_a >= 0) {
          n = prev_node->merge_b - prev_node->merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto& current_node = merged[idx + prev_node->merge_a]; // this tab gives us directly the parent of the node
            
            score_before_merge = merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            //std::cout << current_node << std::endl;
            if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
              new_path.push_back(current_node->l);
            }
            pathsInfo_current.addPathValue(new_path, prev_path_score + score_before_merge - current_node->score, current_node);
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto& current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
            new_path.push_back(current_node->l);
          }
          pathsInfo_current.addPathValue(new_path, prev_path_score + score_no_merge - current_node->score, current_node);
                                                 
        }

      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}

double ForceAlignBeamNoiseStats::getTrueScore(const fl::Variable& output, int b, std::vector<int> pathAnalysis)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data->batch[b];

  auto& hyps = data.hyps;
  auto& merged = data.merged;
  auto& merged_score = data.merged_score;

  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size()-2;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T+1).at(0));
  double score_before_merge;
  double score_no_merge;
  long n;

  for(long t = T; t >= 0; t--) {
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo){
      auto& prev_path = std::get<0>(path_info);
      auto& prev_nodes = std::get<2>(path_info);
      for (auto const& prev_node_info : prev_nodes){
        auto& prev_node = prev_node_info.first;
        auto& prev_path_score = prev_node_info.second;
        if(prev_node->merge_a >= 0) {
          n = prev_node->merge_b - prev_node->merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto& current_node = merged[idx + prev_node->merge_a]; // this tab gives us directly the parent of the node
            
            score_before_merge = merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            //std::cout << current_node << std::endl;
            if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
              new_path.push_back(current_node->l);
            }
            if (new_path.back() == pathAnalysis[pathAnalysis.size() - new_path.size()]){
              pathsInfo_current.addPathValue(new_path, prev_path_score + score_before_merge - current_node->score, current_node);
            }
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto& current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
            new_path.push_back(current_node->l);
          }
          if (new_path.back() == pathAnalysis[pathAnalysis.size() - new_path.size()]){
            pathsInfo_current.addPathValue(new_path, prev_path_score + score_no_merge - current_node->score, current_node);
          }                                    
        }
      }
    }
    //pathsInfo_current.beamIt(beam_size);
  }

  double result;
  for (auto const& path_tuple : pathsInfo_current.pathsInfo){
    auto path = std::get<0>(path_tuple);
    std::reverse(path.begin(), path.end());
    if (path == pathAnalysis){
      result = std::get<1>(path_tuple);
    }
  }
  return result;
}

double ForceAlignBeamNoiseStats::wLER(const fl::Variable& output, fl::Variable& cleantarget, int& beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data->batch;

  int B = data.size();
  std::vector<double> all_wLER(B,0);

#pragma omp parallel for num_threads(B)
  for(int b = 0; b < B; b++) {
    auto paths_weights = extractPathsAndWeightsBackward(output, b, beam_size);
  }
return 0;
}





std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBackwardSimplify(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data_big = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data->batch[b];
  std::cout << "Simplify .... ";
  ForceAlignBeamNoiseStatsDataPtr data;
  simplifyGraph(data_big, data);
  std::cout << "done" << std::endl;

  auto& hyps = data.hyps;
  auto& merged = data.merged;
  auto& merged_score = data.merged_score;


  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size()-2;
  //std::cout << "final ptr " << hyps.at(T+1).at(0) << std::endl;
  pathsInfo_current.addPathValue(void_path, 0., hyps.at(T+1).at(0));
  //std::cout << hyps.at(T+1).at(0) << std::endl;
  double score_before_merge;
  double score_no_merge;
  long n;

  for(long t = T; t >= 0; t--) {
    std::cout << t << "/" << T << std::endl;
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo){
      auto& prev_path = std::get<0>(path_info);
      //auto& prev_path_score = std::get<1>(path_info);
      auto& prev_nodes = std::get<2>(path_info);
      //just a new for here
      for (auto const& prev_node_info : prev_nodes){
        auto& prev_node = prev_node_info.first;
        //std::cout << "Prev node " << prev_node  << " " << prev_node->merge_a << " " << prev_node->merge_b << std::endl;
        auto& prev_path_score = prev_node_info.second;
        if(prev_node->merge_a >= 0) {
          n = prev_node->merge_b - prev_node->merge_a + 1;
          //std::cout << "n " << n <<std::endl;
          for(long idx = 0; idx < n; idx++) {
            auto current_node = merged[idx + prev_node->merge_a]; // this tab gives us directly the parent of the node
            
            score_before_merge = merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            //std::cout << "current " << current_node << std::endl;
            if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
              new_path.push_back(current_node->l);
            }
            pathsInfo_current.addPathValue(new_path, prev_path_score + score_before_merge - current_node->score, current_node);
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
            new_path.push_back(current_node->l);
          }
          pathsInfo_current.addPathValue(new_path, prev_path_score + score_no_merge - current_node->score, current_node);
                                                 
        }

      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}




void ForceAlignBeamNoiseStats::simplifyGraph(ForceAlignBeamNoiseStatsData& data_big, ForceAlignBeamNoiseStatsDataPtr& data_small){
  auto& hyps_big = data_big.hyps;
  auto& merged_big = data_big.merged;
  auto& merged_score_big = data_big.merged_score;

  //ForceAlignBeamNoiseStatsData data_small;
  auto& hyps_small = data_small.hyps;
  auto& merged_small = data_small.merged;
  auto& merged_score_small = data_small.merged_score;


  for(auto & hyps_t : hyps_big) {
    for(auto & node : hyps_t) {
      node.active = false;
    }
  }

  int T = hyps_big.size()-2;
  hyps_big.at(T+1).at(0).active = true;

  for (int t = T+1; t>=1; t--){
    auto& hyps_t = hyps_big.at(t);
    for (auto &node : hyps_t){
      if (node.active){
        if (node.merge_a >= 0){
          int n = node.merge_b - node.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto next_node = merged_big[idx + node.merge_a];
            next_node->active = true;
          }
        } else{
          node.parent->active = true;
        }
      }
    }
  }


  hyps_small.resize(T+2);
  //for (int t=0 ; t<=T+1; t++){
  //  std::vector<ForceAlignBeamNoiseNodeStats> hyp_t;
  //  hyps_small.push_back(hyp_t);
  //}
  //Initalize by duplicating the init node.
  auto& init_node_big = hyps_big.at(0).at(0);
  //auto init_node_small = init_node_big;
  ForceAlignBeamNoiseNodeStats* init_node_small = new ForceAlignBeamNoiseNodeStats(init_node_big);

  //ForceAlignBeamNoiseNodeStats init_node_small = init_node_big;
  //auto& hyps_0_small = hyps_small[0];
  //std::vector<ForceAlignBeamNoiseNodeStats> hyp0 = {init_node_big};
  hyps_small[0].push_back(init_node_small);
  //auto& init_node_small = hyps_small[0].back();
  //hyps_small.at(0).push_back({init_node_small});
  
  std::map<ForceAlignBeamNoiseNodeStats*, ForceAlignBeamNoiseNodeStats*> map_old_new = {{&init_node_big, init_node_small}};
  std::map<ForceAlignBeamNoiseNodeStats*, ForceAlignBeamNoiseNodeStats*> map_prev_old_new = {};

  std::cout << init_node_big.score << " " << init_node_small->score << std::endl;

  for (int t = 1; t<=T+1; t++){
    double score_t_big = -std::numeric_limits<double>::infinity();
    double score_t_small = -std::numeric_limits<double>::infinity();
    //auto& hyps_t_small = hyps_small[t];

    std::cout << t << "/" << T <<std::endl;
    std::swap(map_prev_old_new, map_old_new);
    map_old_new.clear();
    //for (auto& x : map_prev_old_new){
    //  std::cout << x.first << " " << x.second << std::endl;
    //}
    
    auto& hyps_t_big = hyps_big.at(t);
    //std::vector<ForceAlignBeamNoiseNodeStats> hyps_t_small;

    //sort and merge nodes
    
    std::sort(hyps_t_big.begin(), hyps_t_big.end(), [](ForceAlignBeamNoiseNodeStats& a, ForceAlignBeamNoiseNodeStats& b) {
      if (a.active == b.active){
        if(a.l == b.l) { // same as a.lex == b.lex but count the rep1 token
          return a.score > b.score;
        } else {
          return a.l < b.l;
        }
      } else{
        return a.active > b.active;
      }
    });

    auto it_find_not_active = std::find_if(hyps_t_big.begin(), hyps_t_big.end(),
                            [](const ForceAlignBeamNoiseNodeStats& node){
                              return node.active == false;
                            });
    long nhyp;
    if(it_find_not_active == hyps_t_big.end()){
      nhyp = hyps_t_big.size();
    } else{
      nhyp = std::distance(hyps_t_big.begin(), it_find_not_active);
      //std::cout << "nhyp " << nhyp << " " << hyps_t_big[nhyp-1].active << " " << hyps_t_big[nhyp].active << std::endl;
    }

    nhyp = hyps_t_big.size(); // 
                       
    long headidx_big = 0;
    
    std::map<ForceAlignBeamNoiseNodeStats*, double> score_per_merged_node;
    ForceAlignBeamNoiseNodeStats& head_big = hyps_t_big.at(headidx_big);
    ForceAlignBeamNoiseNodeStats* head_small = new ForceAlignBeamNoiseNodeStats(head_big);
    hyps_small[t].push_back(head_small);
    //ForceAlignBeamNoiseNodeStats& head_small = hyps_small[t].back();
    std::cout << head_small << std::endl;
    //std::cout << "1st " << &head_big << " " << head_small << std::endl;
    map_old_new[&head_big] = head_small;
    std::cout << "1.. " << head_big.l << " " << &head_big << " " << head_small << std::endl;

    if (head_big.merge_a >= 0){
      int n = head_big.merge_b - head_big.merge_a + 1;
      for(long idx = 0; idx < n; idx++) {
        auto prev_node_big = merged_big[idx + head_big.merge_a];
        auto prev_merge_score_big = merged_score_big[idx + head_big.merge_a];
        //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
        //  std::cout << "not found 1 " << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
        //}
        score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
      }
    } else{
      auto prev_node_big = head_big.parent;
      auto prev_merge_score_big = head_big.score;
      //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
      //    std::cout << "not found 2" << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
      //}
      score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
    }

    score_t_big = logadd(score_t_big, head_big.score);
    std::cout << "big " << head_big.score << std::endl;
    std::cout << "small " << head_small->score << std::endl;

    for(long h = 1; h < nhyp; h++) {

      auto& elem_big = hyps_t_big.at(h);
      auto& head_big = hyps_t_big.at(headidx_big);
      //std::cout << "h " << elem_big.active << std::endl;
      score_t_big = logadd(score_t_big, elem_big.score);
      //std::cout << "big " << elem_big.score << std::endl;

      
      if(head_big.l == elem_big.l) { //if we have to merge head and elem
        map_old_new[&elem_big] = head_small;
        std::cout << "2.. " << elem_big.l << " " << &elem_big << " " << head_small << std::endl;

        //std::cout << "2st " << &elem_big << " " << head_small << std::endl;
        if (elem_big.merge_a >= 0){
          int n = elem_big.merge_b - elem_big.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto prev_node_big = merged_big[idx + elem_big.merge_a];
            auto prev_merge_score_big = merged_score_big[idx + elem_big.merge_a];
            //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
            //  std::cout << "not found 3"  << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
            //}
            auto it_exist_node = score_per_merged_node.find(map_prev_old_new[prev_node_big]);

            if (it_exist_node == score_per_merged_node.end()){
              score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
            } else{
              it_exist_node->second = logadd(it_exist_node->second, prev_merge_score_big);
            }
            //std::cout << "from " << prev_node_big << " " << prev_merge_score_big << std::endl;

          }
        } else{
          auto prev_node_big = elem_big.parent;
          auto prev_merge_score_big = elem_big.score;
          //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
          //  std::cout << "not found 4" << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
          //}
          auto it_exist_node = score_per_merged_node.find(map_prev_old_new[prev_node_big]);
          if (it_exist_node == score_per_merged_node.end()){
            score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
          } else{
            it_exist_node->second = logadd(it_exist_node->second, prev_merge_score_big);
          }

          //std::cout << "from " << prev_node_big << " " << prev_merge_score_big << std::endl;
        }

        //head_small->score = logadd(head_small->score, elem_big.score);

      } else {
        head_small->merge_a = merged_small.size();
        //std::cout << "NEW HEAD " << std::endl;
        head_small->score = -std::numeric_limits<double>::infinity();
        for (auto const& node_value : score_per_merged_node){
          //std::cout << node_value.first << " " << node_value.second << std::endl;
          head_small->merge_b = merged_small.size();
          merged_small.push_back(node_value.first);
          merged_score_small.push_back(node_value.second);
          head_small->score = logadd(head_small->score, node_value.second);
          std::cout << "small " << node_value.second << std::endl;
          //std::cout << " --> " << node_value.first << " " << node_value.second << std::endl;
        }
        std::cout << "----------> " << head_small->l << " " << head_small->score << std::endl;
        //std::cout << "----------> " << head_small->score << std::endl;
        score_t_small = logadd(score_t_small, head_small->score);
        //int n = head_small->merge_b - head_small->merge_a + 1;
        //std::cout << head_small->merge_a << " " << head_small->merge_b << " n = " << n << std::endl;
        //hyps_t_small.push_back(head_small);

        
        headidx_big = h;

        score_per_merged_node.clear();
        auto& head_big = hyps_t_big.at(headidx_big);
        //ForceAlignBeamNoiseNodeStats* head_small = new ForceAlignBeamNoiseNodeStats(head_big);
        head_small = new ForceAlignBeamNoiseNodeStats(head_big);
        //ForceAlignBeamNoiseNodeStats head_small = head_big;
        hyps_small[t].push_back(head_small);
        //auto& head_small = hyps_small[t].back();
        //std::cout << head_small << std::endl;
        map_old_new[&head_big] = head_small;
        std::cout << "3.. " << head_big.l << " " << &head_big << " " << head_small << std::endl;

        if (head_big.merge_a >= 0){
          int n = head_big.merge_b - head_big.merge_a + 1;
          for(long idx = 0; idx < n; idx++) {
            auto prev_node_big = merged_big[idx + head_big.merge_a];
            auto prev_merge_score_big = merged_score_big[idx + head_big.merge_a];
            //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
            //  std::cout << "not found 5" << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
            //}
            score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
            std::cout << prev_node_big << " " << map_prev_old_new[prev_node_big] << " " << prev_merge_score_big << std::endl;
          }
        } else{
          auto prev_node_big = head_big.parent;
          auto prev_merge_score_big = head_big.score;
          //if(map_prev_old_new.find(prev_node_big) == map_prev_old_new.end()){
          //  std::cout << "not found 6" << head_big.active << " " << prev_node_big->active << " " << prev_merge_score_big << std::endl;
          //}
          score_per_merged_node[map_prev_old_new[prev_node_big]] = prev_merge_score_big;
        }

      }
    }

    //std::cout << "final" << &head_small << &(hyps_t_small.back()) << std::endl;
    //std::cout << "from2 " << head_small << std::endl;
    head_small->merge_a = merged_small.size();
    head_small->score = -std::numeric_limits<double>::infinity();
    //std::cout << "NEW HEAD " << std::endl;
    for (auto const& node_value : score_per_merged_node){
      //std::cout << node_value.first << " " << node_value.second << std::endl;
      head_small->merge_b = merged_small.size();
      merged_small.push_back(node_value.first);
      merged_score_small.push_back(node_value.second);
      head_small->score = logadd(head_small->score, node_value.second);
      std::cout << "small " << node_value.second << std::endl;
      //std::cout << " --> " << node_value.first << " " << node_value.second << std::endl;;
    }
    std::cout << "----------> " << head_small->l << " " << head_small->score << std::endl;
    score_t_small = logadd(score_t_small, head_small->score);
    //int n = head_small->merge_b - head_small->merge_a + 1;
    //std::cout << head_small->merge_a << " " << head_small->merge_b << " n = " << n << std::endl;

    std::cout << score_t_big << " " << score_t_small << std::endl;
  }

}




///////THIS VERSION OF BACKWARD BELLOW IS CURRENTLY THE BEST WE HAVE

/*

struct pathsInfoBeamBackward {
  typedef std::tuple<std::vector<int>, double, ForceAlignBeamNoiseNodeStats*> pathTuple;
  typedef std::tuple<std::vector<int>, double> simplePathTuple;
  std::vector<pathTuple> pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing the adresses of the nodes that lead to the path.
  //             We can have multiple nodes because of the merging operation during the forward pass and because of the multiple alignements.
  

  pathsInfoBeamBackward() {};
  //pathsInfoBeam(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(std::vector<int> path, double value, ForceAlignBeamNoiseNodeStats* node_ptr) {
    // Add the path if not present. Otherwise add the score the already stored path.
    auto it_existing_path = std::find_if(pathsInfo.begin(), pathsInfo.end(),
                          [&path, &node_ptr](const pathTuple& tuple) {return ((std::get<0>(tuple) == path) && (std::get<2>(tuple) == node_ptr));});
    // Verify if the path is present by comparing the first elemnents of the tuples.
    
    if (it_existing_path == pathsInfo.end()) { //if the path is not present, we add it
      //std::vector<ForceAlignBeamNoiseNodeStats*> single_vect{node_ptr};
      pathTuple new_tuple = std::make_tuple(path, value, node_ptr);
      pathsInfo.push_back(new_tuple);
    } else{
      std::get<1>(*it_existing_path) = logadd(std::get<1>(*it_existing_path), value); // Logadd the scores
      //std::get<2>(*it_existing_path).push_back(node_ptr); // Keep track of the node that has generated the path.
    }
  }

  void sortIt(){
    std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
        //return std::get<1>(a) + std::get<2>(a)->score > std::get<1>(b) + std::get<2>(b)->score;
        return std::get<1>(a) + std::get<2>(a)->score > std::get<1>(b) + std::get<2>(b)->score;
      });
  }

  void beamIt(int beam_size){
    sortIt();
    if (beam_size < pathsInfo.size()){ // if the vector is too big.
      // we sort it
      pathsInfo.resize(beam_size);
    }
  }

  std::vector<simplePathTuple> getResult(){ // remove node ptr and reverse paths
    std::vector< simplePathTuple> result;
    for (auto const& path_tuple : pathsInfo){
      auto path = std::get<0>(path_tuple);
      std::reverse(path.begin(), path.end());
      result.push_back(std::make_tuple(path,std::get<1>(path_tuple)));
    }
    return result;
  }

};



std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;


  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size()-2;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T+1).at(0));

  double score_before_merge;
  double score_no_merge;
  long n;

  for(long t = T; t >= 0; t--) {
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo){
      auto& prev_path = std::get<0>(path_info);
      auto& prev_path_score = std::get<1>(path_info);
      auto& prev_node = std::get<2>(path_info);

      if(prev_node->merge_a >= 0) {
        n = prev_node->merge_b - prev_node->merge_a + 1;
        for(long idx = 0; idx < n; idx++) {
          auto& current_node = merged[idx + prev_node->merge_a]; // this tab gives us directly the parent of the node
          
          score_before_merge = merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
            new_path.push_back(current_node->l);
          }
          pathsInfo_current.addPathValue(new_path, prev_path_score + score_before_merge - current_node->score, current_node);
        }
      } else {
        score_no_merge = prev_node->score + prev_node->maxscore;
        auto& current_node = prev_node->parent;

        std::vector<int> new_path = prev_path;
        if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
          new_path.push_back(current_node->l);
        }
        pathsInfo_current.addPathValue(new_path, prev_path_score + score_no_merge - current_node->score, current_node);
                                               
      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}

*/



/*

std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;


  pathsInfoBeamForward pathsInfo_current;
  pathsInfoBeamForward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamForward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size()-2;

  for(std::vector<ForceAlignBeamNoiseNodeStats>& hyps_t : hyps) {
    for(ForceAlignBeamNoiseNodeStats& node : hyps_t) {
      node.gscore = 0.;
    }
  }
  hyps.at(T+1).at(0).gscore = 1.0;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T+1).at(0));

  //long n = fini.merge_b-fini.merge_a+1;
  //pathsInfoBeamBackward pathsInfo_current;
  //pathsInfoBeamBackward pathsInfo_previous;
  //pathsInfo_current = pathsInfoBeamBackward();
  //for(long idx = 0; idx < n; idx++) {
  //  ForceAlignBeamNoiseNodeStats* current_node = merged[idx + fini.merge_a];
  //  std::vector<int> new_path(0); // a void path for now.
  //  pathsInfo_current.addPathValue(new_path, merged_score[idx + fini.merge_a] - current_node->score, current_node);
  //}
  //pathsInfo_current.beamIt(beam_size);

  std::vector<double> sub_merged_gscore;
  std::vector<double> sub_merged_score;
  for(long t = T; t >= 0; t--) {
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamForward();

    //auto paths_by_node = pathsInfo_previous.getPathsByNode();

    for (auto const& prev_node_and_path : pathsInfo_previous.getPathsByNode()){
      auto& prev_node = prev_node_and_path.first;
      auto& prev_paths_info = prev_node_and_path.second;

      //ForceAlignBeamNoiseNodeStats* node = (std::get<2>(path_tuple));
      if(prev_node->merge_a >= 0) {
        long n = prev_node->merge_b - prev_node->merge_a + 1;
        sub_merged_gscore.resize(n);
        sub_merged_score.resize(n);
        std::copy(merged_score.begin()+prev_node->merge_a, merged_score.begin()+prev_node->merge_b+1, sub_merged_score.begin());
        dlogadd(sub_merged_score, sub_merged_gscore, prev_node->gscore);

        for(long idx = 0; idx < n; idx++) {
          auto& current_node = merged[idx + prev_node->merge_a]; // this tab gives us directly the parent of the node
          
          if(sub_merged_score[idx] != -std::numeric_limits<double>::infinity()) {
            current_node->gscore += sub_merged_gscore[idx];
          }
          
          double score_before_merge = merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

          //if(score_before_merge != -std::numeric_limits<double>::infinity()) {
          for (auto const& prev_path_info : prev_paths_info){
            std::vector<int> new_path = std::get<0>(prev_path_info);
            if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
              new_path.push_back(current_node->l);
            }
            pathsInfo_current.addPathValue(new_path, std::get<1>(prev_path_info) + score_before_merge - current_node->score, current_node);
          }
          //}
        }
      } else {
        double score_no_merge = prev_node->score + prev_node->maxscore;
        auto& current_node = prev_node->parent;
        current_node->gscore += prev_node->gscore;

        for (auto const& prev_path_info : prev_paths_info){
          std::vector<int> new_path = std::get<0>(prev_path_info);
          if (new_path.size() == 0 || (new_path.back() != current_node->l && current_node->l != -1)){
            new_path.push_back(current_node->l);
          }
          pathsInfo_current.addPathValue(new_path, std::get<1>(prev_path_info) + score_no_merge - current_node->score, current_node);
        }                                                 
      }
    }
    pathsInfo_current.beamIt(beam_size, true);
  }
  return pathsInfo_current.getResult(true);
}

*/

std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBoth(const fl::Variable& output, int b, int beam_size)
{
    std::tuple<std::vector< std::tuple<std::vector<int>, double>>, std::vector< std::tuple<std::vector<int>, double>>> res_ford_back;

#pragma omp parallel for num_threads(2)
  for (int i=0; i<2; i++){
    if (i==0){
      std::get<0>(res_ford_back) = this->extractPathsAndWeightsForward(output, b, beam_size);
    } else{
      std::get<1>(res_ford_back) = this->extractPathsAndWeightsBackward(output, b, beam_size);
    }
  }
  auto& res_ford = std::get<0>(res_ford_back);
  auto& res_back = std::get<1>(res_ford_back);

  double tot_score_ford = -std::numeric_limits<double>::infinity();
  for (auto& path_weight : res_ford){
    tot_score_ford = logadd(tot_score_ford, std::get<1>(path_weight));
  }

  double tot_score_back = -std::numeric_limits<double>::infinity();
  for (auto& path_weight : res_back){
    tot_score_back = logadd(tot_score_back, std::get<1>(path_weight));
  }

  std::vector< std::tuple<std::vector<int>, double>> res_merge = res_ford;

  for (auto& tuple_back : res_back){
    auto it_path_merge_found = std::find_if(res_merge.begin(), res_merge.end(),
                          [&tuple_back](const std::tuple<std::vector<int>, double>& tuple_merge) {
                            return (std::get<0>(tuple_merge) == std::get<0>(tuple_back));
                          });
    if (it_path_merge_found == res_merge.end()){
      res_merge.push_back(tuple_back);
    } else{
      if (std::get<1>(tuple_back) > std::get<1>(*it_path_merge_found)){
         std::get<1>(*it_path_merge_found) = std::get<1>(tuple_back);
      }
    }
  }

  std::sort(res_merge.begin(), res_merge.end(), [](std::tuple<std::vector<int>, double>& a, std::tuple<std::vector<int>, double>& b) {
        return std::get<1>(a) > std::get<1>(b);
  });

  double tot_score_merge = -std::numeric_limits<double>::infinity();
  for (auto& path_weight : res_merge){
    tot_score_merge = logadd(tot_score_merge, std::get<1>(path_weight));
  }

  std::cout << "Ford: " << tot_score_ford << ", Back: " << tot_score_back << ", Both: " << tot_score_merge << std::endl;

  return res_merge;
}


std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::greedyPath(const fl::Variable& output, int b)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data->batch[b];
  int T = data.hyps.size()-2;

  std::vector<int> path_v(T);

  ForceAlignBeamNoiseNodeStats *node = &(data.hyps.at(T).at(0));
  int t = T;
  while(node && (node->l >= 0)) {
    path_v[--t] = node->l;
    node = node->parent;
  }

  //cleanGreedyPath;
  int idx = 0;
  for(int t = 0; t < T; t++) {
    if(path_v[t] >= 0) {
      if(idx != t) {
        path_v[idx] = path_v[t];
      }
      idx++;
    }
  }
  w2l::uniq(path_v);
  std::vector< std::tuple<std::vector<int>, double>> result = {std::make_tuple(path_v, 0)};

  return result;
}



/*
std::vector< std::tuple<std::vector<int>, double>> ForceAlignBeamNoiseStats::extractPathsAndWeightsBackward(const fl::Variable& output, int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  assert(hyps.size() >= 3);
  const size_t T = hyps.size()-2;
  for(auto & hyps_t : hyps) {
    for(auto & node : hyps_t) {
      node.active = false;
    }
  }

  auto& fini = hyps.at(T+1).at(0);
  fini.active = true;

  long n = fini.merge_b-fini.merge_a+1;
  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  for(long idx = 0; idx < n; idx++) {
    ForceAlignBeamNoiseNodeStats* current_node = merged[idx + fini.merge_a];
    std::vector<int> new_path(0); // a void path for now.
    pathsInfo_current.addPathValue(new_path, merged_score[idx + fini.merge_a] - current_node->score, current_node);
  }
  pathsInfo_current.beamIt(beam_size);

  for(long t = T; t >= 0; t--) {

    pathsInfo_previous = pathsInfo_current;

    pathsInfo_current = pathsInfoBeamBackward();
    for (auto& path_tuple : pathsInfo_previous.pathsInfo){ // x has a ForceAlignBeamNoiseNodeStats* key and std::vector<std::tuple<std::vector<int>, double>> value

      ForceAlignBeamNoiseNodeStats* node = (std::get<2>(path_tuple));
      if(node->merge_a >= 0) {
        long n = node->merge_b - node->merge_a + 1;
        for(long idx = 0; idx < n; idx++) {
          auto& current_node = merged[idx + node->merge_a]; // this tab gives us directly the parent of the node
          auto& current_score = merged_score[idx + node->merge_a];
          if(current_score != -std::numeric_limits<double>::infinity()) {
            std::vector<int> new_path = std::get<0>(path_tuple);
            if (new_path.size() == 0 || (new_path.back() != node->l && node->l != -1)){
              new_path.push_back(node->l);
            }
            pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + current_score + node->maxscore
                                                                - current_node->score, current_node);
          }
        }
      } else {
        auto& current_score = node->score;
        auto& current_node = node;

        std::vector<int> new_path = std::get<0>(path_tuple); //x.first is the key, the path
        if (new_path.size() == 0 || (new_path.back() != node->l && node->l != -1)){
          new_path.push_back(node->l);
        }
        pathsInfo_current.addPathValue(new_path, std::get<1>(path_tuple) + current_score + node->maxscore
                                                               - current_node->score, current_node->parent);                                                      
      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}
*/

void ForceAlignBeamNoiseStats::backward(std::vector<fl::Variable>& inputs, const fl::Variable& goutput, std::shared_ptr<ForceAlignBeamNoiseStatsBatchData> data, bool is_noisytarget)
{
  auto& emissions = inputs[0];
  auto& transitions = inputs[1];
  auto& noisytarget = inputs[2];
  auto& knoisytarget = inputs[3];

  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  const int B = emissions.dims(2);

  std::vector<float> gemissions_v(emissions.elements(), 0);
  std::vector<float> gtransitions_v(B*transitions.elements(), 0);
  std::vector<float> goutput_v(goutput.elements());
  goutput.host(goutput_v.data());

#pragma omp parallel for num_threads(B)
  for(int b = 0; b < B; b++) {
    auto gemissions_p = gemissions_v.data() + b*N*T;
    auto gtransitions_p = gtransitions_v.data() + b*N*N;
    double gscore = goutput_v[b];

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    //noiselm_.zeroGrad();

    if(merged.size() != merged_score.size()) {
      std::cout << "$ merged scores have wrong sizes" << std::endl;
      throw std::invalid_argument("merged scores have wrong sizes");
    }

    for(std::vector<ForceAlignBeamNoiseNodeStats>& hyps_t : hyps) {
      for(ForceAlignBeamNoiseNodeStats& node : hyps_t) {
        node.gscore = 0;
        node.active = false;
      }
    }
    hyps.at(T+1).at(0).active = true;
    hyps.at(T+1).at(0).gscore = gscore;
 
    std::vector<double> sub_merged_score;
    std::vector<double> sub_merged_gscore;
    for(long t = T; t >= 0; t--) {
      for(ForceAlignBeamNoiseNodeStats& node : hyps.at(t+1)) {
        if(node.active) {
          if(node.merge_a >= 0) {
            long n = node.merge_b-node.merge_a+1;
            sub_merged_score.resize(n);
            sub_merged_gscore.resize(n);

            std::copy(merged_score.begin()+node.merge_a, merged_score.begin()+node.merge_b+1, sub_merged_score.begin());

            dlogadd(sub_merged_score, sub_merged_gscore, node.gscore);
            for(long idx = 0; idx < n; idx++) {
              if(sub_merged_score[idx] != -std::numeric_limits<double>::infinity()) {
                accnode(node, *merged.at(node.merge_a+idx), gemissions_p, gtransitions_p, t, T, N, noiselm_, sub_merged_gscore[idx]);
              }
            }
          } else {
            accnode(node, *node.parent, gemissions_p, gtransitions_p, t, T, N, noiselm_, node.gscore);
          }
        }
      }
    }
  }

  // reduce
  for(int b = 1; b < B; b++) {
    auto gtransitions_p = gtransitions_v.data();
    auto gtransitions_b_p = gtransitions_v.data() + b*N*N;
    for(int k = 0; k < N*N; k++) {
      gtransitions_p[k] += gtransitions_b_p[k];
    }
  }

  emissions.addGrad(fl::Variable(af::array(N, T, B, gemissions_v.data()), false));
  transitions.addGrad(fl::Variable(af::array(N, N, 1, gtransitions_v.data()), false));
  noiselm_.backward();
}


static void cleanViterbiPath(int *path_p, int64_t T)
{
  int64_t idx = 0;
  for(int64_t t = 0; t < T; t++) {
    if(path_p[t] >= 0) {
      if(idx != t) {
        path_p[idx] = path_p[t];
      }
      idx++;
    }
  }
  for(int64_t t = idx; t < T; t++) {
    path_p[t] = -1;
  }
}

af::array ForceAlignBeamNoiseStats::viterbi(const fl::Variable& output)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;
  int64_t B = data->batch.size();
  int64_t T = data->batch[0].hyps.size()-2;

  std::vector<int> path_v(T*B);

  for(int64_t b = 0; b < B; b++) {
    auto path_p = path_v.data() + b*T;
    ForceAlignBeamNoiseNodeStats *node = &(data->batch[b].hyps.at(T).at(0));
    int64_t t = T;
    while(node && (node->l >= 0)) {
      path_p[--t] = node->l;
      node = node->parent;
    }
    cleanViterbiPath(path_p, T);  
  }
  return af::array(T, B, path_v.data());
}

af::array ForceAlignBeamNoiseStats::viterbiWord(const fl::Variable& output)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;
  int64_t B = data->batch.size();
  int64_t T = data->batch[0].hyps.size()-2;

  std::vector<int> path_v(T*B);

  for(int64_t b = 0; b < B; b++) {
    auto path_p = path_v.data() + b*T;
    ForceAlignBeamNoiseNodeStats *node = &(data->batch[b].hyps.at(T).at(0));
    long t = T;
    while(node && (node->l >= 0)) {
      if(node->key) {
        path_p[--t] = node->key->lm;
      } else {
        path_p[--t] = -1;
      }
      node = node->parent;
    }
    cleanViterbiPath(path_p, T);
  }
  return af::array(T, B, path_v.data());
}

ForceAlignBeamNoiseNodeStats aggregateStatsNodes(std::vector<ForceAlignBeamNoiseNodeStats>& nodes){
  ForceAlignBeamNoiseNodeStats result;
  //first find the best node and aggregate on it if mode = "weighted"
  
  int idx_best = 0;
  int current_idx = 0;
  double best_score = -1e200;
  for (ForceAlignBeamNoiseNodeStats& node : nodes){
    if (node.score > best_score){
      idx_best = current_idx;
      best_score = node.score;
    }
    current_idx += 1;
  }
  result = nodes[idx_best];
  result.sum_coeff = 1;

  current_idx = 0;
  double coeff;
  double head_score_ini = result.score;
  for (ForceAlignBeamNoiseNodeStats& node : nodes){
    if (current_idx != idx_best){
      coeff = exp(node.score - head_score_ini);
      result.mergeWStats(node, coeff);
    }
  }
  result.finalizeWStats();

  return result;
}

void displayNoiseModel(NoiseLMLetterSwapUnit& noiselm, bool withInsDel, w2l::Dictionary* keys = nullptr){
  int width=6;
  int prec=1;
  int nb_token = keys->entrySize();
  std::cout << std::endl  << "Noise model " << std::endl;
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
  double display;
  for (int i=0 ; i<nb_token; i++){
    if (keys != nullptr){
      std::cout << std::setw(1) << keys->getEntry(i);     
    } else{
      std::cout << std::setw(1) << "";
    }
    std::cout << " |";
    for (int j=0 ; j<nb_token; j++){
      display = exp(noiselm.scoreSwap(i,j)) * 100;
      std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
    }
    if (withInsDel){ //Ins column
      std::cout << " |";
        display = exp(noiselm.scoreInsertion(i)) * 100;
        std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display; 
    }
    std::cout << std::endl;
  }

  std::cout << fill << std::endl;
  if (withInsDel){ // Del row
    std::cout << "   ";
    for (int i=0 ; i<nb_token; i++){
      display = exp(noiselm.scoreDeletion(i)) * 100;
      std::cout << std::setw(width) << std::setprecision(prec) << std::fixed << display;
    }
    std::cout << std::endl;
  }

  std::cout << std::setw(0) << std::setprecision(6) << std::defaultfloat; //get back to default display option
}
