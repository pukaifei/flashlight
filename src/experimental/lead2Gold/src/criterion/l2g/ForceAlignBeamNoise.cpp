#include "experimental/lead2Gold/src/criterion/l2g/ForceAlignBeamNoise.h"

// for w2l::gettargetSize()
#include <assert.h>
#include <algorithm>
#include <iomanip>
#include "common/FlashlightUtils.h"
#include "criterion/CriterionUtils.h"

static double logadd(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity() &&
      b == -std::numeric_limits<double>::infinity()) {
    return a;
  }
  if (a > b) {
    return a + log1p(exp(b - a));
  }
  return b + log1p(exp(a - b));
}
/*
static void printhypsfast(long t, std::vector<ForceAlignBeamNoiseNode>& hyps,
long nhyp)
{
  std::cout << "FST
===============================================================" << std::endl;
  double norm = 0;
  for(long h = 0; h < nhyp; h++) {
    ForceAlignBeamNoiseNode& node = hyps[h];
    std::cout << "[" << h << "] t=" << t << " l=" << node.l << " score=" <<
node.score << " noisytarget_t=" << node.noisytarget_t << std::endl; norm +=
node.score*node.score;
  }
  std::cout << "norm=" << norm << std::endl;
}
*/
ForceAlignBeamNoise::ForceAlignBeamNoise(
    w2l::Dictionary& tokenDict,
    std::shared_ptr<NoiseTrie> lex,
    NoiseLMLetterSwapUnit& noiselm,
    long B,
    double threshold,
    int top_k,
    bool count_noise,
    bool count_noise_sort)
    : tokenDict_(tokenDict),
      lex_(lex),
      noiselm_(noiselm),
      B_(B),
      threshold_(threshold),
      top_k_(top_k),
      count_noise_(count_noise),
      count_noise_sort_(count_noise_sort) {}

/*
void ForceAlignBeamNoise::statsAccDelta(double delta)
{
  stats_.at(0) += 1;
  stats_.at(1) += fabs(delta);
  stats_.at(2) += fabs(delta)*fabs(delta);
}

void ForceAlignBeamNoise::statsAccMaxScore(double maxscore)
{
  stats_.at(3) += 1;
  stats_.at(4) += fabs(maxscore);
}

void ForceAlignBeamNoise::clearStats()
{
  stats_.resize(5);
  for(size_t i = 0; i < stats_.size(); i++) {
    stats_.at(i) = 0;
  }
}

std::vector<double>& ForceAlignBeamNoise::stats()
{
  return stats_;
}
*/
struct Comp { // to sort idx for top k feature
  Comp(const float* v) : _v(v) {}
  bool operator()(int a, int b) {
    return _v[a] > _v[b];
  }
  const float* _v;
};

fl::Variable ForceAlignBeamNoise::forward(
    fl::Variable& emissions,
    fl::Variable& transitions,
    fl::Variable& noisytarget,
    fl::Variable& knoisytarget) {
  return ForceAlignBeamNoise::forward(
      emissions, emissions, transitions, noisytarget, knoisytarget);
}

fl::Variable ForceAlignBeamNoise::forward(
    fl::Variable& emissions,
    fl::Variable& emissions_forsort,
    fl::Variable& transitions,
    fl::Variable& noisytarget,
    fl::Variable& knoisytarget) {
  // emissions = fl::logSoftmax(emissions, 0);
  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  const int B = emissions.dims(2);
  const int mS = noisytarget.dims(0);
  const int mkS = knoisytarget.dims(0);

  // af::print("transitions", transitions.array());
  // af::print("emissions", emissions.array());

  int top_k = top_k_ <= 0
      ? N
      : std::min(std::max(2, top_k_), N); // we don't allow top_k = 1

  // fl::Variable scaling;
  // if (noiselm_.autoScale() == true) {
  //  scaling = fl::log(fl::sum(fl::exp(emissions),{0}));
  //} else {
  //  scaling =
  //  fl::Variable((af::constant(noiselm_.scaleValue(),1,T,B,1)),false);
  //}

  // std::vector<float> scaling_v(scaling.elements());
  std::vector<float> emissions_v(emissions.elements());
  std::vector<float> emissions_forsort_v(emissions_forsort.elements());
  std::vector<float> transitions_v(transitions.elements());
  std::vector<int> noisytarget_v(noisytarget.elements());
  std::vector<int> knoisytarget_v(knoisytarget.elements());
  emissions.host(emissions_v.data());
  emissions_forsort.host(emissions_forsort_v.data());
  transitions.host(transitions_v.data());
  noisytarget.host(noisytarget_v.data());
  knoisytarget.host(knoisytarget_v.data());
  // scaling.host(scaling_v.data());

  auto data = std::make_shared<ForceAlignBeamNoiseBatchData>();
  data->batch.resize(B);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    auto emissions_p = emissions_v.data() + b * N * T;
    auto emissions_forsort_p = emissions_forsort_v.data() + b * N * T;
    auto noisytarget_p = noisytarget_v.data() + b * mS;
    auto knoisytarget_p = knoisytarget_v.data() + b * mkS;
    // auto scaling_p = scaling_v.data() + b*T;
    const int S = w2l::getTargetSize(noisytarget_p, mS);
    const int kS = w2l::getTargetSize(knoisytarget_p, mkS);

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    // Debug purpose. Print info
    // std::cout << "ktarget: " << std::endl;
    // for (int i=0 ; i < kS ; i++){
    //  std::cout << knoisytarget_p[i] << " ";
    //}
    // std::cout << std::endl;

    //

    // +root +end
    hyps.resize(T + 2);
    std::vector<double> maxscores(T), maxscores_forsort(T);

    NoiseTrieNode* lexroot;
    if (lex_) {
      lexroot = lex_->root(); // start with the token "|" as root
    }

    hyps.at(0).emplace_back(
        tokenDict_.getIndex("|"),
        tokenDict_.getIndex("|"),
        0,
        0,
        0,
        -1,
        -1,
        nullptr,
        lexroot,
        -1);
    // hyps.at(0).push_back({-1, 0,  -1,  -1, nullptr, keytrieroot->child(0),
    // nullptr, -1, 0,0,0});

    // std::cout << "sould be 0: " << *hyps.at(0).back().nb_token_ptr <<
    // std::endl;
    double noisescore;
    double base_logProbNoise;
    double logProbNoise;
    double score, score_forsort;
    double baseScore, baseScore_forsort;
    long noiselmstate;
    int nb_remaining_target;
    int nb_remaining_frame;
    NoiseTrieNode* lexNode = nullptr; // useless ?
    // NoiseTrieNode *letter = nullptr;
    long tok;
    std::vector<ForceAlignBeamNoiseNode> newhyps;

    std::vector<int> idx_unsort(N), idx_sorted(N);
    std::iota(idx_unsort.begin(), idx_unsort.end(), 0);
    idx_sorted = idx_unsort;
    int K;

    for (long t = 0; t < T; t++) {
      newhyps.clear();
      // std::cout << "T " << t << std::endl;
      if (!lex_ && top_k < N) { // prune top k
        idx_sorted = idx_unsort;
        std::partial_sort(
            idx_sorted.begin(),
            idx_sorted.begin() + top_k,
            idx_sorted.end(),
            Comp(emissions_forsort_p + t * N));
        // std::partial_sort( idx_sorted.begin(), idx_sorted.begin() + top_k,
        // idx_sorted.end(), Comp(emissions_p + t*N) );
      }
      // if (b==3){
      //  std::cout << "t: " << t << " size hyp: " << hyps.at(t).size() <<
      //  std::endl;
      //}
      for (ForceAlignBeamNoiseNode& prev : hyps.at(t)) {
        // add previous token if not in top k

        // std::cout << "prev.l: " << prev.l << " prev target: " <<
        // knoisytarget_p[prev.knoisytarget_t ] << std::endl;
        if (lex_) {
          idx_sorted.clear();
          for (auto child : prev.lexNode->children()) {
            if (child) {
              if (prev.tok == child->idx()) {
                idx_sorted.push_back(tokenDict_.getIndex("1"));
              } else {
                idx_sorted.push_back(child->idx());
              }
            }
          }
          // if (b==3){
          //  std::cout << "add prev: " << prev.tokNet << std::endl;
          //}
          idx_sorted.push_back(prev.tokNet);
          if (!prev.lexNode->labels().empty()) {
            idx_sorted.push_back(tokenDict_.getIndex("|"));
          }
          // apply top_k
          if (top_k < idx_sorted.size()) { // should not be the common case
            std::partial_sort(
                idx_sorted.begin(),
                idx_sorted.begin() + top_k,
                idx_sorted.end(),
                Comp(emissions_forsort_p + t * N));
          }
          K = std::min(top_k, (int)idx_sorted.size());

        } else { // add the previous token if not present if we don't use a
                 // lexicon
          K = top_k;
          if (top_k < N && prev.tokNet >= 0 &&
              (std::find(
                   idx_sorted.begin(),
                   idx_sorted.begin() + top_k,
                   prev.tokNet) == idx_sorted.begin() + top_k)) {
            idx_sorted[K] = prev.tokNet;
            K++;
          }
        }
        // if (b==3){
        //  std::cout << "K: " << K << std::endl;
        //  for( int idx = 0; idx < K; idx++ ) {
        //    std::cout << idx_sorted[idx] << " ";
        //  }
        //  std::cout << std::endl;
        //  for( int idx = 0; idx < idx_sorted.size(); idx++ ) {
        //    std::cout << idx_sorted[idx] << " ";
        //  }
        //  std::cout << std::endl;
        //}
        for (int idx = 0; idx < K; idx++) {
          int& i = idx_sorted[idx];

          tok = (i == tokenDict_.getIndex("1")) ? prev.tok : i;
          if (lex_) {
            lexNode = (tok == tokenDict_.getIndex("|"))
                ? lexroot
                : prev.lexNode->child(tok);
          }

          if (t == 0) { // generate the first hypothesis except the rep label
            if (i != tokenDict_.getIndex("1")) {
              if (std::isfinite(noiselm_.scoreSwap(
                      knoisytarget_p[prev.knoisytarget_t + 1],
                      tok)) // If the next letter can be swapped
                  || std::isfinite(noiselm_.scoreDeletion(
                         tok)) // if the next letter can be deleted
                  ||
                  (std::isfinite(noiselm_.scoreInsertion(
                       knoisytarget_p[prev.knoisytarget_t + 1])) &&
                   (prev.knoisytarget_t + 2 <
                    kS) // if the next letter can be inserted
                   &&
                   std::isfinite(noiselm_.scoreSwap(
                       knoisytarget_p[prev.knoisytarget_t + 2],
                       tok)) // and then the next one can be swapped or deleted
                             // (useless to verify since already done)
                   )) {
                baseScore = prev.score + emissions_p[t * N + i];
                baseScore_forsort =
                    prev.score_forsort + emissions_forsort_p[t * N + i];
                newhyps.emplace_back(
                    i,
                    tok,
                    baseScore,
                    baseScore_forsort,
                    0,
                    prev.noisytarget_t + 1,
                    prev.knoisytarget_t + 1,
                    &prev,
                    lexNode,
                    noiselmstate);
              }
            }
          } else {
            baseScore = prev.score + emissions_p[t * N + i] +
                transitions_v[i * N + prev.tokNet];
            baseScore_forsort = prev.score_forsort +
                emissions_forsort_p[t * N + i] +
                transitions_v[i * N + prev.tokNet];

            nb_remaining_target = kS - (prev.noisytarget_t + 1);
            nb_remaining_frame = T - (t + 1);
            // if (b==3){
            //  std::cout << "remain target: " << nb_remaining_target <<
            //  std::endl; std::cout << "remain frame: " << nb_remaining_frame
            //  << std::endl; std::cout << "prev.knoisytarget_t " <<
            //  prev.knoisytarget_t << std::endl; std::cout << "kS " << kS <<
            //  std::endl; std::cout << "prev tok " << prev.tokNet << " i: " <<
            //  i << std::endl;
            //}
            if (prev.tokNet == i) {
              // we force to follow the target if we don't have enough frames to
              // finish it. Except for 1 letter if we allow insertion.
              // std::cout << "nb_remaining_frame " << nb_remaining_frame << "
              // nb_remaining_target " << nb_remaining_target << std::endl;
              // std::cout << "follow target: " << std::endl;
              if (nb_remaining_frame >=
                  nb_remaining_target) { // THE PROBLEM IS HERE
                if (nb_remaining_frame - 1 >= nb_remaining_target ||
                    (std::isfinite(noiselm_.scoreSwap(
                        knoisytarget_p[prev.knoisytarget_t],
                        tok))) // If the next letter can be swapped
                    ||
                    (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                     (nb_remaining_frame - 1 >=
                      nb_remaining_target)) // if the next letter can be deleted
                    ||
                    (std::isfinite(noiselm_.scoreInsertion(
                         knoisytarget_p[prev.knoisytarget_t])) &&
                     (prev.knoisytarget_t + 2 <
                      kS) // if the next letter can be inserted
                     &&
                     std::isfinite(noiselm_.scoreSwap(
                         knoisytarget_p[prev.knoisytarget_t + 1],
                         tok)) // and then the next one can be swapped or
                               // deleted (useless to verify since already done)
                     )) {
                  // newhyps.push_back({i, baseScore, prev.noisytarget_t,
                  // prev.knoisytarget_t, &prev, prev.letter, key, noiselmstate,
                  // prev.nb_subs, prev.nb_ins, prev.nb_del}); std::cout <<
                  // "same" << std::endl; std::cout << "add hyp: " << std::endl;
                  newhyps.emplace_back(
                      i,
                      prev.tok,
                      baseScore,
                      baseScore_forsort,
                      0,
                      prev.noisytarget_t,
                      prev.knoisytarget_t,
                      &prev,
                      prev.lexNode,
                      noiselmstate);
                  // std::cout << "newhyps: " << newhyps.size() << std::endl;
                }
              }
            } else {
              // or we change the letter, hence the key

              if (noiselm_.allowInsertion() &&
                  prev.knoisytarget_t + 2 <
                      kS) { // a letter has been added to the noisy
                            // transcription. +1 only because we can
                if (std::isfinite(noiselm_.scoreSwap(
                        knoisytarget_p[prev.knoisytarget_t + 2],
                        tok)) // If the next letter can be swapped
                    || (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                        (nb_remaining_frame - 1 >= nb_remaining_target -
                             2)) // if the next letter can be deleted
                    ||
                    (std::isfinite(noiselm_.scoreInsertion(
                         knoisytarget_p[prev.knoisytarget_t + 2])) &&
                     (prev.knoisytarget_t + 4 <
                      kS) // if the next letter can be inserted
                     &&
                     std::isfinite(noiselm_.scoreSwap(
                         knoisytarget_p[prev.knoisytarget_t + 3],
                         tok)) // and then the next one can be swapped or
                               // deleted (useless to verify since already done)
                     )) {
                  logProbNoise = noiselm_.scoreInsertion(
                                     knoisytarget_p[prev.knoisytarget_t]) +
                      noiselm_.scoreSwap(
                          knoisytarget_p[prev.knoisytarget_t + 1], prev.tok);
                  noisescore = noiselm_.scale_noise() * logProbNoise +
                      noiselm_.tkn_score();
                  // newhyps.push_back({i, baseScore + noisescore,
                  // prev.noisytarget_t + 2, prev.knoisytarget_t + 2, &prev,
                  // letter, key, noiselmstate, prev.nb_subs + 1, prev.nb_ins +
                  // 1, prev.nb_del});
                  if (std::isfinite(noisescore)) {
                    score = count_noise_ ? baseScore + noisescore : baseScore;
                    score_forsort = count_noise_sort_
                        ? baseScore_forsort + noisescore
                        : baseScore_forsort;
                    newhyps.emplace_back(
                        i,
                        tok,
                        score,
                        score_forsort,
                        logProbNoise,
                        prev.noisytarget_t + 2,
                        prev.knoisytarget_t + 2,
                        &prev,
                        lexNode,
                        noiselmstate);
                  }
                }
              }

              base_logProbNoise = noiselm_.allowInsertion()
                  ? noiselm_.scoreNoInsertion()
                  : 0.; // no letter is inserted
              if (noiselm_.allowDeletion() &&
                  prev.knoisytarget_t < kS) { // for now allow only 1 deletion
                                              // and one last deletion
                // Verify that the proposed hyp is probable
                if (std::isfinite(noiselm_.scoreSwap(
                        knoisytarget_p[prev.knoisytarget_t],
                        tok)) // If the next letter can be swapped
                    ||
                    (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                     (nb_remaining_frame - 1 >=
                      nb_remaining_target)) // if the next letter can be deleted
                    ||
                    (std::isfinite(noiselm_.scoreInsertion(
                         knoisytarget_p[prev.knoisytarget_t])) &&
                     (prev.knoisytarget_t + 2 <
                      kS) // if the next letter can be inserted
                     &&
                     std::isfinite(noiselm_.scoreSwap(
                         knoisytarget_p[prev.knoisytarget_t + 1],
                         tok)) // and then the next one can be swapped or
                               // deleted (useless to verify since already done)
                     )) {
                  // we allow a deletion only if we have enough frames to finish
                  // the sentence.
                  if (nb_remaining_frame >= nb_remaining_target) {
                    logProbNoise =
                        base_logProbNoise + noiselm_.scoreDeletion(prev.tok);
                    noisescore = noiselm_.scale_noise() * logProbNoise +
                        noiselm_.tkn_score();
                    // newhyps.push_back({i, baseScore + noisescore,
                    // prev.noisytarget_t, prev.knoisytarget_t, &prev, letter,
                    // key, noiselmstate, prev.nb_subs, prev.nb_ins, prev.nb_del
                    // + 1});
                    if (std::isfinite(noisescore)) {
                      score = count_noise_ ? baseScore + noisescore : baseScore;
                      score_forsort = count_noise_sort_
                          ? baseScore_forsort + noisescore
                          : baseScore_forsort;
                      newhyps.emplace_back(
                          i,
                          tok,
                          score,
                          score_forsort,
                          logProbNoise,
                          prev.noisytarget_t,
                          prev.knoisytarget_t,
                          &prev,
                          lexNode,
                          noiselmstate);
                    }
                  }
                }
              }

              if (noiselm_.allowSwap() && prev.knoisytarget_t + 1 < kS) {
                // Verify that the proposed hyp is probable. Otherwise most of
                // the beam hyps are useless if the noise model is sparse.
                if (std::isfinite(noiselm_.scoreSwap(
                        knoisytarget_p[prev.knoisytarget_t + 1],
                        tok)) // If the next letter can be swapped
                    || (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                        (nb_remaining_frame - 1 >= nb_remaining_target -
                             1)) // if the next letter can be deleted
                    ||
                    (std::isfinite(noiselm_.scoreInsertion(
                         knoisytarget_p[prev.knoisytarget_t + 1])) &&
                     (prev.knoisytarget_t + 3 <
                      kS) // if the next letter can be inserted
                     &&
                     std::isfinite(noiselm_.scoreSwap(
                         knoisytarget_p[prev.knoisytarget_t + 2],
                         tok)) // and then the next one can be swapped or
                               // deleted (useless to verify since already done)
                     )) {
                  logProbNoise = base_logProbNoise +
                      noiselm_.scoreSwap(
                          knoisytarget_p[prev.knoisytarget_t], prev.tok);
                  noisescore = noiselm_.scale_noise() * logProbNoise +
                      noiselm_.tkn_score();
                  // std::cout << "noisescore " <<
                  // knoisytarget_p[prev.knoisytarget_t] << "->" <<
                  // prev.letter->idx() << " " << noisescore << std::endl;
                  // newhyps.push_back({i, baseScore + noisescore,
                  // prev.noisytarget_t + 1, prev.knoisytarget_t + 1, &prev,
                  // letter, key, noiselmstate, prev.nb_subs + 1, prev.nb_ins,
                  // prev.nb_del}); std::cout << "swap" << std::endl;
                  if (std::isfinite(noisescore)) {
                    score = count_noise_ ? baseScore + noisescore : baseScore;
                    score_forsort = count_noise_sort_
                        ? baseScore_forsort + noisescore
                        : baseScore_forsort;
                    newhyps.emplace_back(
                        i,
                        tok,
                        score,
                        score_forsort,
                        logProbNoise,
                        prev.noisytarget_t + 1,
                        prev.knoisytarget_t + 1,
                        &prev,
                        lexNode,
                        noiselmstate);
                  }
                }
              }
            }
          }
        }
      }

      if (newhyps.size() == 0) {
        throw std::invalid_argument("No hypothesis generated");
      }

      // offset scores
      double maxscore = -std::numeric_limits<double>::infinity();
      double maxscore_forsort = -std::numeric_limits<double>::infinity();
      for (ForceAlignBeamNoiseNode& hyp : newhyps) {
        maxscore = std::max(maxscore, hyp.score);
        maxscore_forsort = std::max(maxscore_forsort, hyp.score_forsort);
      }
      for (ForceAlignBeamNoiseNode& hyp : newhyps) {
        hyp.score -= maxscore;
        hyp.score_forsort -= maxscore_forsort;
        hyp.maxscore = maxscore; // useless now ? pass pointeur ?
      }
      maxscores[t] = maxscore;
      maxscores_forsort[t] = maxscore_forsort;

      // if (b==3){
      //  std::cout << "before prune " << newhyps.size() << std::endl;
      //}
      // prune hyps
      if (threshold_ > 0) {
        float npruned = 0;
        for (size_t i = 0; i < newhyps.size(); i++) {
          if (newhyps.at(i).score_forsort > maxscore_forsort - threshold_) {
            if (i != npruned) {
              newhyps.at(npruned) = newhyps.at(i);
            }
            npruned++;
          }
        }
        newhyps.resize(npruned);
      }
      if (newhyps.size() == 0) {
        throw std::invalid_argument("All hypothesis are pruned");
      }
      // if (b==3){
      //  std::cout << "after prune " << newhyps.size() << std::endl;
      //}

      // merge identical nodes
      std::sort(
          newhyps.begin(),
          newhyps.end(),
          [](ForceAlignBeamNoiseNode& a, ForceAlignBeamNoiseNode& b) {
            if (a.lexNode == b.lexNode) {
              if (a.tokNet ==
                  b.tokNet) { // same as a.lex == b.lex but count the rep1 token
                if (a.knoisytarget_t == b.knoisytarget_t) {
                  return a.score_forsort > b.score_forsort;
                } else {
                  return a.noisytarget_t < b.noisytarget_t;
                }
              } else {
                return a.tokNet < b.tokNet;
              }
            } else {
              return a.lexNode < b.lexNode;
            }
          });

      long headidx = 0;
      long nhyp = newhyps.size();
      // if (b==3){
      //  std::cout << "nhyp before merge " << nhyp << std::endl;
      //}
      // double coeff;
      // double sum_coeff = 1.0;
      // double head_score_ini = newhyps.at(headidx).score;

      for (long h = 1; h < nhyp; h++) {
        ForceAlignBeamNoiseNode& elem = newhyps.at(h);
        ForceAlignBeamNoiseNode& head = newhyps.at(headidx);

        if ((head.lexNode == elem.lexNode) && (head.tokNet == elem.tokNet) &&
            (head.knoisytarget_t ==
             elem.knoisytarget_t)) { // maybe to change when key level
          if (head.merge_a < 0) {
            head.merge_a = merged.size();
            merged.push_back(head.parent); /* note: parent is in hyps */
            merged_score.push_back(head.score);
          }
          head.merge_b = merged.size();
          merged.push_back(elem.parent); /* note: parent is in hyps */
          merged_score.push_back(elem.score);
          // std::cout << "head.score " << head.score << std::endl;
          // std::cout << "elem.score " << elem.score << std::endl;
          head.score = logadd(head.score, elem.score);
          head.score_forsort = logadd(head.score_forsort, elem.score_forsort);
          // std::cout << "head.score " << head.score << std::endl;
          // coeff = exp(elem.score - head_score_ini);
          // head.nb_subs += elem.nb_subs * coeff;
          // head.nb_ins += elem.nb_ins * coeff;
          // head.nb_del += elem.nb_del * coeff;
          // sum_coeff += coeff;
        } else {
          // head.nb_subs /= sum_coeff;
          // head.nb_ins /= sum_coeff;
          // head.nb_del /= sum_coeff;
          headidx++;
          if (headidx != h) {
            newhyps.at(headidx) = newhyps.at(h);
          }
          // sum_coeff = 1.0;
          // head_score_ini = newhyps.at(headidx).score;
        }
      }
      nhyp =
          std::min(headidx + 1, nhyp); // should be headidx+1 when it is
                                       // working. To handle the case nhyp = 0
      // if (b==3){
      //  std::cout << "nhyp after " << nhyp << std::endl;
      //}
      //
      // beam it
      std::sort(
          newhyps.begin(),
          newhyps.begin() + nhyp,
          [](ForceAlignBeamNoiseNode& a, ForceAlignBeamNoiseNode& b) {
            return a.score_forsort > b.score_forsort;
          });
      // std::sort(newhyps.begin(), newhyps.begin()+nhyp,
      // [](ForceAlignBeamNoiseNode& a, ForceAlignBeamNoiseNode& b) { return
      // a.score > b.score; });
      nhyp = std::min(nhyp, B_);
      hyps.at(t + 1).insert(
          hyps.at(t + 1).end(), newhyps.begin(), newhyps.begin() + nhyp);
      // for(ForceAlignBeamNoiseNode& hyp : hyps.at(t+1)) {
      //  std::cout << " merge final score " << hyp.score << std::endl;
      //}
    }
    // for(ForceAlignBeamNoiseNode& hyp : hyps.at(T)) {
    //    std::cout << "s " << hyp.score << std::endl;
    //    std::cout << "M " << hyp.maxscore << std::endl;
    //}
    // finish
    //-1e220 -> we assign a probability even if the beam failed. To change,
    //maybe it causes a bug ForceAlignBeamNoiseNode fini(-1, -1e200, -1, -1,
    // nullptr, nullptr, nullptr, -1);
    ForceAlignBeamNoiseNode fini("fini");
    for (ForceAlignBeamNoiseNode& prev : hyps.at(T)) {
      if (fini.merge_a < 0) {
        fini.merge_a = merged.size();
        fini.merge_b = merged.size() - 1;
      }

      double score, score_forsort;
      double noisescore;
      double base_noisescore = 0.;
      // onlyHypWithTargetFinished is deprecated
      bool merge_it = false;
      // auto nb_subs = prev.nb_subs;
      // auto nb_ins = prev.nb_ins;
      // auto nb_del = prev.nb_del;
      if (prev.knoisytarget_t + 1 == kS &&
          noiselm_.allowSwap()) { // no insertion + swap current + no insertion
                                  // after
        base_logProbNoise =
            noiselm_.allowInsertion() ? 2.0 * noiselm_.scoreNoInsertion() : 0.;
        logProbNoise = base_logProbNoise +
            noiselm_.scoreSwap(knoisytarget_p[prev.knoisytarget_t], prev.tok);
        noisescore =
            noiselm_.scale_noise() * logProbNoise + noiselm_.tkn_score();
        merge_it = true;
      } else if (
          prev.knoisytarget_t + 2 == kS &&
          noiselm_
              .allowInsertion()) { // insertion + swap current + no insertion
        base_logProbNoise = noiselm_.scoreNoInsertion() +
            noiselm_.scoreInsertion(knoisytarget_p[prev.knoisytarget_t]);
        logProbNoise = noiselm_.allowSwap() ? base_logProbNoise +
                noiselm_.scoreSwap(
                    knoisytarget_p[prev.knoisytarget_t + 1], prev.tok)
                                            : base_logProbNoise;
        noisescore = noiselm_.allowSwap()
            ? noiselm_.scale_noise() * logProbNoise + noiselm_.tkn_score()
            : noiselm_.scale_noise() * logProbNoise;
        merge_it = true;
      } else if (
          prev.knoisytarget_t == kS &&
          noiselm_.allowDeletion()) { // deletion of the previous letter + no
                                      // insertion after
        logProbNoise = noiselm_.allowInsertion()
            ? noiselm_.scale_noise() * noiselm_.scoreNoInsertion()
            : 0.;
        noisescore =
            noiselm_.scale_noise() * logProbNoise + noiselm_.tkn_score();
        merge_it = true;
      }
      if (merge_it) {
        // prev.score += noisescore;
        // prev.score_bis = prev.score;
        score = count_noise_ ? prev.score + noisescore : prev.score;
        score_forsort = count_noise_sort_ ? prev.score_forsort + noisescore
                                          : prev.score_forsort;
        // score_forsort = prev.score_forsort + noisescore;
        fini.merge_b = merged.size();
        merged.push_back(&prev);
        merged_score.push_back(score);
        // std::cout << fini.score << " " << score << std::endl;
        fini.score = logadd(fini.score, score);
        fini.score_forsort = logadd(fini.score_forsort, score_forsort);
        // fini.score_bis = fini.score;
        // std::cout << fini.score << std::endl;

        int n = fini.merge_b - fini.merge_a + 1;
        // fini.nb_subs = ((n-1) * fini.nb_subs + nb_subs) / n;
        // fini.nb_ins = ((n-1) * fini.nb_ins + nb_ins) / n;
        // fini.nb_del = ((n-1) * fini.nb_del + nb_del) / n;
      }
    }

    // std::cout << "FIN " << fini.score << std::endl;

    double sum_of_maxscores = 0.0;
    double sum_of_maxscores_forsort = 0.0;
    // std::cout << "MAXSCORES " << std::endl;
    // int k=1;
    for (double& score : maxscores) {
      sum_of_maxscores += score;
    }
    for (double& score : maxscores_forsort) {
      sum_of_maxscores_forsort += score;
    }
    // std::cout << "MAX ";
    // std::cout << sum_of_maxscores << std::endl;

    if (std::isfinite(sum_of_maxscores)) {
      fini.score += sum_of_maxscores;
      // fini.score_bis = fini.score;
    }
    if (std::isfinite(sum_of_maxscores_forsort)) {
      fini.score_forsort += sum_of_maxscores_forsort;
      // fini.score_bis = fini.score;
    }
    hyps.at(T + 1).push_back(std::move(fini));
    // std::cout << "FIN 2 " << fini.score << std::endl;
  }

  std::vector<float> result_p(B);
  for (int b = 0; b < B; b++) {
    result_p[b] = data->batch[b].hyps.at(T + 1).at(0).score;
  }
  auto result = af::array(B, result_p.data());
  auto grad_func = [this, data](
                       std::vector<fl::Variable>& inputs,
                       const fl::Variable& goutput) {
    this->backward(inputs, goutput, data);
  };

  std::vector<fl::Variable> inputs = {
      emissions, transitions, noisytarget, knoisytarget};
  // const std::vector<fl::Variable> noiselmParams = noiselm_.params();
  // inputs.insert(inputs.end(), noiselmParams.begin(), noiselmParams.end());

  return fl::Variable(
      result,
      inputs,
      grad_func,
      std::make_shared<ForceAlignBeamNoiseVariablePayload>(data));
}

// This is not working
/*
fl::Variable ForceAlignBeamNoise::transformOutput(fl::Variable& beam_transform,
fl::Variable& emissions_previous, fl::Variable& emissions_new, fl::Variable&
transitions, fl::Variable& noisytarget, fl::Variable& knoisytarget)
{
  auto payload = beam_transform.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;

  const int N = emissions_new.dims(0);
  const int T = emissions_new.dims(1);
  const int B = emissions_new.dims(2);

  std::vector<float> emissions_previous_v(emissions_previous.elements());
  std::vector<float> emissions_new_v(emissions_new.elements());
  emissions_previous.host(emissions_previous_v.data());
  emissions_new.host(emissions_new_v.data());


#pragma omp parallel for num_threads(B)
  for(int b = 0; b < B; b++) {
    auto emissions_previous_p = emissions_previous_v.data() + b*N*T;
    auto emissions_new_p = emissions_new_v.data() + b*N*T;


    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;


    double sum_maxscore_new = 0;
    for(long t = 1; t <= T; t++) {
      //std::cout << "t " << t << " max old " << hyps.at(t)[0].maxscore <<
std::endl; double maxscore_new = -std::numeric_limits<double>::infinity();
      for(ForceAlignBeamNoiseNode& node : hyps.at(t)) {
        //if (t== 5 || t==6){
        //  std::cout << "before old " << node.score_bis << " new " <<
node.score << std::endl;
          //std::cout << emissions_previous_p[(t-1)*N+node.l] << " " <<
emissions_new_p[(t-1)*N+node.l] << std::endl;
        //}
        //
        //std::cout << "new node " << std::endl;
        //

        if(node.merge_a >= 0) {
          //if (t== 5 || t==6){
          //  std::cout << "merged ! " << std::endl;
          //}
            long n = node.merge_b-node.merge_a+1;
            for(long idx = 0; idx < n; idx++) {
              merged_score[idx + node.merge_a] += node.maxscore
                                                  - (merged[idx +
node.merge_a]->score_bis + emissions_previous_p[(t-1)*N+node.l]) //remove old
prev score and emission
                                                  + merged[idx +
node.merge_a]->score + emissions_new_p[(t-1)*N+node.l]; maxscore_new =
std::max(maxscore_new, merged_score[idx + node.merge_a]);
              //std::cout << "idx = " << idx << std::endl;
              //std::cout << merged[idx + node.merge_a]->score_bis << " " <<
merged[idx + node.merge_a]->score << std::endl;
              //if (t== 5 || t==6){
              //  std::cout << "idx = " << idx << std::endl;
              //  std::cout << merged[idx + node.merge_a]->score_bis << " " <<
merged[idx + node.merge_a]->score << std::endl;
              //}
            }

        } else {

          node.score += node.maxscore
                      - (node.parent->score_bis +
emissions_previous_p[(t-1)*N+node.l])
                      + node.parent->score + emissions_new_p[(t-1)*N+node.l];

          maxscore_new = std::max(maxscore_new, node.score);

          //std::cout << node.parent->score_bis << " " << node.parent->score <<
std::endl;
          //std::cout << emissions_previous_p[(t-1)*N+node.l] << " " <<
emissions_new_p[(t-1)*N+node.l] << std::endl; maxscore_new =
std::max(maxscore_new, node.score);
        }
      }
      //std::cout << "t " << t << " max new " << maxscore_new << std::endl;
      sum_maxscore_new += maxscore_new;
      //offset maxscore
      for(ForceAlignBeamNoiseNode& node : hyps.at(t)) {
        node.maxscore = maxscore_new;
        if(node.merge_a >= 0) {
            long n = node.merge_b - node.merge_a + 1;
            double score_node = -std::numeric_limits<double>::infinity();
            for(long idx = 0; idx < n; idx++) {
              merged_score[idx + node.merge_a] -= maxscore_new;
              score_node = logadd(score_node, merged_score[idx + node.merge_a]);
            }
            node.score = score_node;

        } else {
          node.score -= maxscore_new;
        }
      }
      std::sort(hyps.at(t).begin(), hyps.at(t).end(),
[](ForceAlignBeamNoiseNode& a, ForceAlignBeamNoiseNode& b) { return a.score >
b.score; });
    }

    //correction of the final node
    //
    auto& final_node = hyps.at(T+1).at(0);
    double score_final = -std::numeric_limits<double>::infinity();
    if(final_node.merge_a >= 0) { //always true
      long n = final_node.merge_b - final_node.merge_a + 1;
      for(long idx = 0; idx < n; idx++) {
        merged_score[idx + final_node.merge_a] = merged_score[idx +
final_node.merge_a]
                                            - merged[idx +
final_node.merge_a]->score_bis
                                            + merged[idx +
final_node.merge_a]->score; score_final = logadd(score_final, merged_score[idx +
final_node.merge_a]);
      }
    }

    //std::cout << "score_final " << score_final<<std::endl;
    //std::cout << "sum " << sum_maxscore_new<<std::endl;
    final_node.score = score_final + sum_maxscore_new;
    //std::cout << "final old " << final_node.score_bis << "final new " <<
final_node.score << std::endl;
  }



  //Reconstruct the flashlight Variable with a good grad func and inputs.
  std::vector<float> result_p(B);
  for(int b = 0; b < B; b++) {
    result_p[b] = data->batch[b].hyps.at(T+1).at(0).score;
  }
  auto result = af::array(B, result_p.data());
  auto grad_func = [this,data](
    std::vector<fl::Variable>& inputs,
    const fl::Variable& goutput) {
    this->backward(inputs, goutput, data);
  };

  std::vector<fl::Variable> inputs = {emissions_new, transitions, noisytarget,
knoisytarget}; const std::vector<fl::Variable> noiselmParams =
noiselm_.params(); inputs.insert(inputs.end(), noiselmParams.begin(),
noiselmParams.end());

  return fl::Variable(result, inputs, grad_func,
std::make_shared<ForceAlignBeamNoiseVariablePayload>(data));

}
*/

static void
dlogadd(std::vector<double>& score, std::vector<double>& gscore, double g) {
  double m = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < score.size(); i++) {
    m = std::max(m, score[i]);
  }
  double sum = 0;
  for (size_t i = 0; i < score.size(); i++) {
    sum += exp(score[i] - m);
  }
  for (size_t i = 0; i < score.size(); i++) {
    if (m ==
        -std::numeric_limits<double>::infinity()) { // when no hypothesis have
                                                    // been found. actually
                                                    // should no occurs...
      gscore[i] == 0.0;
    } else {
      gscore[i] = exp(score[i] - m) / sum * g;
    }
  }
}

static void accnode(
    ForceAlignBeamNoiseNode& node,
    ForceAlignBeamNoiseNode& prev,
    float* gemissions_p,
    float* gtransitions_p,
    float* gscale_noise_p,
    long t,
    long T,
    long N,
    NoiseLMLetterSwapUnit& noiselm,
    double g) {
  if (t >= 0 && t < T) {
    gemissions_p[t * N + node.tokNet] += g;
    if (prev.tokNet >= 0) {
      gtransitions_p[node.tokNet * N + prev.tokNet] += g;
    }
    // to change later!
    if (node.logProbNoise != 0) {
      //  noiselm.accGradScore(node.lm, g);
      gscale_noise_p[0] += node.logProbNoise * g;
    }
  }
  prev.gscore += g;
  prev.active = true;
}
/*
static void print_node(std::ostream& f, long t, ForceAlignBeamNoiseNode& node)
{
  if(t >= 0) {
    f << "node_" << t << "_" << (long)(&node);
  } else {
    f << "node_R" << "_" << (long)(&node);
  }
}

static void define_node(std::ostream& f, long t, ForceAlignBeamNoiseNode& node)
{
  print_node(f, t, node);
  f << " [label_l=\"" << node.l << "\"  label_t=\"" << node.knoisytarget_t <<
"\"];" << std::endl;
}

static void connect_nodes(std::ostream& f, long t1, ForceAlignBeamNoiseNode&
node1, long t2, ForceAlignBeamNoiseNode& node2)
{
  print_node(f, t1, node1);
  f << " -> ";
  print_node(f, t2, node2);
  f << ";" << std::endl;
}
*/
// void ForceAlignBeamNoise::showEditOps(const fl::Variable& output, int64_t b)
//{
//  auto payload = output.getPayload();
//  if(!payload) {
//    throw std::invalid_argument("expecting a payload on provided Variable");
//  }
//  auto data =
//  std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;
//  auto& hyps = data->batch[b].hyps;
//  fini = hyps.at(T+1).at(0);
//
//  std::cout <<
//}
/*
void ForceAlignBeamNoise::showTarget(const fl::Variable& output, int64_t b,
std::ostream& f)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;
  //auto& merged_noisytarget_score = data->batch[b].merged_noisytarget_score;

  f << "digraph decfal {" << std::endl;
  long T = hyps.size()-2;
  for(std::vector<ForceAlignBeamNoiseNode>& hyps_t : hyps) {
    for(ForceAlignBeamNoiseNode& node : hyps_t) {
      node.gscore = 0;
      node.active = false;
    }
  }
  hyps.at(T+1).at(0).active = true;
  define_node(f, T, hyps.at(T+1).at(0));

  std::vector<double> sub_merged_score;
  for(long t = T; t >= 0; t--) {
    for(ForceAlignBeamNoiseNode& node : hyps.at(t+1)) {
      if(node.active) {
        if(node.merge_a >= 0) {
          long n = node.merge_b-node.merge_a+1;
          sub_merged_score.resize(n);
          std::copy(merged_score.begin()+node.merge_a,
merged_score.begin()+node.merge_b+1, sub_merged_score.begin()); for(long idx =
0; idx < n; idx++) { if(sub_merged_score[idx] !=
-std::numeric_limits<double>::infinity()) {
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
*/
// implement threshold, resolve memory issues, max_nb_paths param
// don't sort every time, change the data structure
struct pathsInfo {
  std::map<std::vector<int>, double> pathsToValue; // paths
  std::multimap<double, decltype(pathsToValue.begin())> reverseMap;
  int max_nb_paths = 50;
  double threshold = 1e-8;
  pathsInfo(){};
  pathsInfo(std::vector<int> path, double value) {
    pathsToValue[path] = value;
  };

  void addPathValue(std::vector<int> path, double value) {
    auto it_existing_path = pathsToValue.find(path);
    if (it_existing_path ==
        pathsToValue.end()) { // if path is not present, we may add it
      // if (value >= threshold){
      if (pathsToValue.size() <
          max_nb_paths) { // if we have enough space to add a new path
        auto it = pathsToValue.insert({path, value})
                      .first; // create a new path and add value
        reverseMap.insert({value, it});
      } else { // else we have to remove to worse one if the current is better
        // auto it_min = min_element(pathsToValue.begin(), pathsToValue.end(),
        //                [](decltype(pathsToValue)::value_type& l,
        //                decltype(pathsToValue)::value_type& r) -> bool {
        //                return l.second < r.second; });

        /*auto it_min = min_element(pathsToValue.begin(), pathsToValue.end(),
                        [](auto& l, auto& r) { return l.second < r.second; });
                        */

        auto it_min = reverseMap.begin();

        if (it_min->first <
            value) { // if the lowest one is worse than the new added value, we
                     // remove it and add the new.
          pathsToValue.erase(
              it_min
                  ->second); // erase the corresponding iterator in pathsToValue
          reverseMap.erase(it_min); // And erase it in the multimap.
          auto it = pathsToValue.insert({path, value})
                        .first; // create a new path and add value
          reverseMap.insert({value, it});
        }
      }
      //}
    } else {
      double old_value = it_existing_path->second;

      auto range = reverseMap.equal_range(old_value);
      auto it = reverseMap.begin();
      for (auto i = range.first; i != range.second; ++i) {
        if (i->second == it_existing_path) {
          it = i;
          break;
        }
      }
      reverseMap.erase(it);
      it_existing_path->second = logadd(it_existing_path->second, value);
      reverseMap.insert({it_existing_path->second, it_existing_path});
    }
  }
};
/*
std::map< std::vector<int>, double>
ForceAlignBeamNoise::extractPathsAndWeights(const fl::Variable& output, int64_t
b)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;

  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  assert(hyps.size() >= 3);
  const size_t T = hyps.size()-2;
  int i=0;
  for(auto & hyps_t : hyps) {
    for(auto & node : hyps_t) {
      node.gscore = 0;
      node.active = false;
    }
    i++;
  }
  auto& fini = hyps.at(T+1).at(0);
  fini.active = true;
  std::unordered_map<ForceAlignBeamNoiseNode*, pathsInfo> nodeToPaths;
  //std::vector<double> sub_merged_score;
  //std::vector<ForceAlignBeamNoiseNode*> sub_merged;

  long n = fini.merge_b-fini.merge_a+1;

  //sub_merged_score.resize(n);
  //std::copy(merged_score.begin()+fini.merge_a,
merged_score.begin()+fini.merge_b+1, sub_merged_score.begin());

  //sub_merged.resize(n);
  //std::copy(merged.begin()+fini.merge_a, merged.begin()+fini.merge_b+1,
sub_merged.begin());

  for(long idx = 0; idx < n; idx++) {
    // auto* node_previous = &fini;
    ForceAlignBeamNoiseNode* node_current = merged[idx + fini.merge_a];
    //if(!node_current->active) {
    node_current->active = true;
    //}
    auto & pathsInfo_current = nodeToPaths[node_current];
    std::vector<int> new_path(1, (int)node_current->l); // replace with letter
    pathsInfo_current.addPathValue(new_path, merged_score[idx + fini.merge_a] -
node_current->score); //should be equal to the noise added at the end
    //nodeToPaths[(long)(node_current)] = pathsInfo_current;
  }


  for(long t = T-1; t >= 0; t--) {
    for(ForceAlignBeamNoiseNode& node : hyps.at(t+1)) {
      if(node.active) {
        auto &pathsInfo_previous = nodeToPaths[&node];
        if(node.merge_a >= 0) {
          long n = node.merge_b-node.merge_a+1;


          //sub_merged_score.resize(n);
          //std::copy(merged_score.begin()+node.merge_a,
merged_score.begin()+node.merge_b+1, sub_merged_score.begin());

          //sub_merged.resize(n);
          //std::copy(merged.begin()+node.merge_a,
merged.begin()+node.merge_b+1, sub_merged.begin());


          for(long idx = 0; idx < n; idx++) {
            auto node_current = merged[idx + node.merge_a];
            const auto& node_score_previous = merged_score[idx + node.merge_a];

            if(node_score_previous != -std::numeric_limits<double>::infinity())
{
              //if(!node_current->active) {
              node_current->active = true;
              //}

              auto &pathsInfo_current = nodeToPaths[node_current];

              for (auto const& x : pathsInfo_previous.pathsToValue){
                std::vector<int> new_path = x.first; //x.first is the key, the
path

                if (new_path.back() != node_current->l && node_current->l !=
-1){ new_path.push_back(node_current->l);
                }
                pathsInfo_current.addPathValue(new_path, x.second +
node_score_previous + node.maxscore
                                                                -
node_current->score);
              }
              //nodeToPaths[(long)(node_current)] = pathsInfo_current;
            }
          }
        } else {
          auto& node_score_previous = node.score;
          auto& node_current = node.parent;
          //if(!node_current->active) {
          node_current->active = true;
          //}

          auto &pathsInfo_current = nodeToPaths[node_current];
          for (auto const& x : pathsInfo_previous.pathsToValue){
            std::vector<int> new_path = x.first; //x.first is the key, the path
            if (new_path.back() != node_current->l && node_current->l != -1){
              new_path.push_back(node_current->l);
            }
            pathsInfo_current.addPathValue(new_path, x.second +
node_score_previous + node.maxscore
                                                                -
node_current->score);
          }
          //nodeToPaths[(long)(node_current)] = pathsInfo_current;
        }
      }
      nodeToPaths.erase(&node);
    }



  }
  auto pathsInfo = nodeToPaths[&hyps.at(0).at(0)];
  return pathsInfo.pathsToValue;
}
*/

void ForceAlignBeamNoise::backward(
    std::vector<fl::Variable>& inputs,
    const fl::Variable& goutput,
    std::shared_ptr<ForceAlignBeamNoiseBatchData> data) {
  auto& emissions = inputs[0];
  auto& transitions = inputs[1];
  auto& noisytarget = inputs[2];
  auto& knoisytarget = inputs[3];
  // auto& noiselmParam = inputs[4];

  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  const int B = emissions.dims(2);

  std::vector<float> gemissions_v(emissions.elements(), 0);
  std::vector<float> gtransitions_v(B * transitions.elements(), 0);
  std::vector<float> gscale_noise_v(B, 0);
  std::vector<float> goutput_v(goutput.elements());
  goutput.host(goutput_v.data());

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    auto gemissions_p = gemissions_v.data() + b * N * T;
    auto gtransitions_p = gtransitions_v.data() + b * N * N;
    auto gscale_noise_p = gscale_noise_v.data() + b;
    double gscore = goutput_v[b];

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    // noiselm_.zeroGrad();
    if (merged.size() != merged_score.size()) {
      std::cout << "$ merged scores have wrong sizes" << std::endl;
      throw std::invalid_argument("merged scores have wrong sizes");
    }

    for (std::vector<ForceAlignBeamNoiseNode>& hyps_t : hyps) {
      for (ForceAlignBeamNoiseNode& node : hyps_t) {
        node.gscore = 0;
        node.active = false;
      }
    }
    hyps.at(T + 1).at(0).active = true;
    hyps.at(T + 1).at(0).gscore = gscore;

    std::vector<double> sub_merged_score;
    std::vector<double> sub_merged_gscore;
    for (long t = T; t >= 0; t--) {
      for (ForceAlignBeamNoiseNode& node : hyps.at(t + 1)) {
        if (node.active) {
          if (node.merge_a >= 0) {
            long n = node.merge_b - node.merge_a + 1;
            sub_merged_score.resize(n);
            sub_merged_gscore.resize(n);
            std::copy(
                merged_score.begin() + node.merge_a,
                merged_score.begin() + node.merge_b + 1,
                sub_merged_score.begin()); // could remove copy
            dlogadd(sub_merged_score, sub_merged_gscore, node.gscore);
            for (long idx = 0; idx < n; idx++) {
              if (sub_merged_score[idx] !=
                  -std::numeric_limits<double>::infinity()) {
                accnode(
                    node,
                    *merged.at(node.merge_a + idx),
                    gemissions_p,
                    gtransitions_p,
                    gscale_noise_p,
                    t,
                    T,
                    N,
                    noiselm_,
                    sub_merged_gscore[idx]);
              }
            }
          } else {
            accnode(
                node,
                *node.parent,
                gemissions_p,
                gtransitions_p,
                gscale_noise_p,
                t,
                T,
                N,
                noiselm_,
                node.gscore);
          }
        }
      }
    }
  }

  // reduce
  for (int b = 1; b < B; b++) {
    // reduce transition
    auto gtransitions_p = gtransitions_v.data();
    auto gtransitions_b_p = gtransitions_v.data() + b * N * N;
    for (int k = 0; k < N * N; k++) {
      gtransitions_p[k] += gtransitions_b_p[k];
    }

    // reduce scale_noise
    auto gscale_noise_p = gscale_noise_v.data();
    auto gscale_noise_b_p = gscale_noise_v.data() + b;
    gscale_noise_p[0] += gscale_noise_b_p[0];
  }

  emissions.addGrad(
      fl::Variable(af::array(N, T, B, gemissions_v.data()), false));
  transitions.addGrad(
      fl::Variable(af::array(N, N, 1, gtransitions_v.data()), false));
  // noiselmParam.addGrad(fl::Variable(af::array(1, gscale_noise_v.data()),
  // false));
}

static void cleanViterbiPath(int* path_p, int T) {
  int idx = 0;
  for (int t = 0; t < T; t++) {
    if (path_p[t] >= 0) {
      if (idx != t) {
        path_p[idx] = path_p[t];
      }
      idx++;
    }
  }
  for (int t = idx; t < T; t++) {
    path_p[t] = -1;
  }
}

af::array ForceAlignBeamNoise::viterbi(const fl::Variable& output) {
  auto payload = output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
      std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)
          ->data;
  int64_t B = data->batch.size();
  int64_t T = data->batch[0].hyps.size() - 2;

  std::vector<int> path_v(T * B);

  for (int64_t b = 0; b < B; b++) {
    auto path_p = path_v.data() + b * T;
    ForceAlignBeamNoiseNode* node = &(data->batch[b].hyps.at(T).at(0));
    int64_t t = T;
    while (node && (node->tokNet >= 0)) {
      path_p[--t] = node->tokNet;
      node = node->parent;
    }
    //    cleanViterbiPath(path_p, T);
  }
  return af::array(T, B, path_v.data());
}

/*
af::array ForceAlignBeamNoise::viterbiWord(const fl::Variable& output)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data;
  int64_t B = data->batch.size();
  int64_t T = data->batch[0].hyps.size()-2;

  std::vector<int> path_v(T*B);

  for(int64_t b = 0; b < B; b++) {
    auto path_p = path_v.data() + b*T;
    ForceAlignBeamNoiseNode *node = &(data->batch[b].hyps.at(T).at(0));
    long t = T;
    while(node && (node->tokNet >= 0)) {
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
*/

struct pathsInfoBeamBackward {
  typedef std::tuple<
      std::vector<int>,
      double,
      std::map<ForceAlignBeamNoiseNode*, double>,
      double>
      pathTuple;
  typedef std::tuple<std::vector<int>, double> simplePathTuple;
  std::vector<pathTuple> pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing
  //  the adresses of the nodes that have lead to the path.
  //             We can have multiple nodes because of the merging operation
  //             during the forward pass and because of the multiple
  //             alignements.

  pathsInfoBeamBackward(){};
  // pathsInfoBeam(std::vector<int> path, double value,
  // ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(
      std::vector<int> path,
      double value,
      ForceAlignBeamNoiseNode* node_ptr) {
    // Add the path if not present. Otherwise add the score of the already
    // stored path.
    auto it_existing_path = std::find_if(
        pathsInfo.begin(), pathsInfo.end(), [&path](const pathTuple& tuple) {
          return (std::get<0>(tuple) == path);
        });
    // Verify if the path is present by comparing the first elemnents of the
    // tuples.

    if (it_existing_path ==
        pathsInfo.end()) { // if the path is not present, we add it
      std::map<ForceAlignBeamNoiseNode*, double> single_map = {
          {node_ptr, value}};
      pathTuple new_tuple =
          std::make_tuple(path, value, single_map, node_ptr->score);
      pathsInfo.push_back(new_tuple);
    } else {
      std::get<1>(*it_existing_path) = logadd(
          std::get<1>(*it_existing_path),
          value); // total score of the path currently

      auto& node_map = std::get<2>(*it_existing_path);
      auto it_exist_node = node_map.find(node_ptr);
      if (it_exist_node == node_map.end()) {
        node_map[node_ptr] = value;
        // std::get<3>(*it_existing_path) =
        // logadd(std::get<3>(*it_existing_path), node_ptr->score);
        // std::get<3>(*it_existing_path) =
        // std::max(std::get<3>(*it_existing_path), node_ptr->score);
      } else {
        it_exist_node->second = logadd(it_exist_node->second, value);
      }
    }
  }

  void sortIt() {
    // we first compute the sorting criteria
    //

    for (auto& path_info : pathsInfo) {
      double res = -std::numeric_limits<double>::infinity();
      for (auto& node_value : std::get<2>(path_info)) {
        auto& node_ptr = node_value.first;
        auto& value = node_value.second;
        res = logadd(res, node_ptr->score + value);
        // res = std::max(res, node_ptr->score + value);
      }
      std::get<3>(path_info) = res;
    }

    std::sort(
        pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b) {
          // return std::get<1>(a) + std::get<2>(a)->score > std::get<1>(b) +
          // std::get<2>(b)->score; return std::get<1>(a) + std::get<3>(a) >
          // std::get<1>(b) + std::get<3>(b);
          return std::get<3>(a) > std::get<3>(b);
        });
  }

  void beamIt(int beam_size) {
    sortIt();
    if (beam_size < pathsInfo.size()) { // if the vector is too big.
      // we sort it
      pathsInfo.resize(beam_size);
    }
  }

  std::vector<simplePathTuple>
  getResult() { // remove node ptr and reverse paths
    std::vector<simplePathTuple> result;
    for (auto const& path_tuple : pathsInfo) {
      auto path = std::get<0>(path_tuple);
      std::reverse(path.begin(), path.end());
      result.push_back(std::make_tuple(path, std::get<1>(path_tuple)));
    }
    return result;
  }
};

std::vector<std::tuple<std::vector<int>, double>>
ForceAlignBeamNoise::extractPathsAndWeightsBackward(
    const fl::Variable& output,
    int b,
    int beam_size) {
  auto payload = output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data =
      std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)
          ->data->batch[b];

  auto& hyps = data.hyps;
  auto& merged = data.merged;
  auto& merged_score = data.merged_score;

  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size() - 2;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T + 1).at(0));
  // std::cout << &hyps.at(T+1).at(0) << std::endl;
  double score_before_merge;
  double score_no_merge;
  long n;

  for (long t = T; t >= 0; t--) {
    // std::cout << t << "/" << T << std::endl;
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo) {
      auto& prev_path = std::get<0>(path_info);
      // auto& prev_path_score = std::get<1>(path_info);
      auto& prev_nodes = std::get<2>(path_info);
      // just a new for here
      for (auto const& prev_node_info : prev_nodes) {
        auto& prev_node = prev_node_info.first;
        auto& prev_path_score = prev_node_info.second;
        if (prev_node->merge_a >= 0) {
          n = prev_node->merge_b - prev_node->merge_a + 1;
          for (long idx = 0; idx < n; idx++) {
            auto& current_node =
                merged[idx + prev_node->merge_a]; // this tab gives us directly
                                                  // the parent of the node

            score_before_merge =
                merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            // std::cout << current_node << std::endl;
            if (new_path.size() == 0 ||
                (new_path.back() != current_node->tokNet && t != 0)) {
              new_path.push_back(current_node->tokNet);
            }
            pathsInfo_current.addPathValue(
                new_path,
                prev_path_score + score_before_merge - current_node->score,
                current_node);
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto& current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 ||
              (new_path.back() != current_node->tokNet && t != 0)) {
            new_path.push_back(current_node->tokNet);
          }
          pathsInfo_current.addPathValue(
              new_path,
              prev_path_score + score_no_merge - current_node->score,
              current_node);
        }
      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}

double ForceAlignBeamNoise::getTrueScore(
    const fl::Variable& output,
    int b,
    std::vector<int> pathAnalysis) {
  auto payload = output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data =
      std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)
          ->data->batch[b];

  auto& hyps = data.hyps;
  auto& merged = data.merged;
  auto& merged_score = data.merged_score;

  pathsInfoBeamBackward pathsInfo_current;
  pathsInfoBeamBackward pathsInfo_previous;
  pathsInfo_current = pathsInfoBeamBackward();
  std::vector<int> void_path(0); // a void path for the final node.
  int T = hyps.size() - 2;

  pathsInfo_current.addPathValue(void_path, 0., &hyps.at(T + 1).at(0));
  double score_before_merge;
  double score_no_merge;
  long n;

  for (long t = T; t >= 0; t--) {
    std::swap(pathsInfo_previous, pathsInfo_current);
    pathsInfo_current = pathsInfoBeamBackward();

    for (auto const& path_info : pathsInfo_previous.pathsInfo) {
      auto& prev_path = std::get<0>(path_info);
      auto& prev_nodes = std::get<2>(path_info);
      for (auto const& prev_node_info : prev_nodes) {
        auto& prev_node = prev_node_info.first;
        auto& prev_path_score = prev_node_info.second;
        if (prev_node->merge_a >= 0) {
          n = prev_node->merge_b - prev_node->merge_a + 1;
          for (long idx = 0; idx < n; idx++) {
            auto& current_node =
                merged[idx + prev_node->merge_a]; // this tab gives us directly
                                                  // the parent of the node

            score_before_merge =
                merged_score[idx + prev_node->merge_a] + prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            // std::cout << current_node << std::endl;
            if (new_path.size() == 0 ||
                (new_path.back() != current_node->tokNet && t != 0)) {
              new_path.push_back(current_node->tokNet);
            }
            if (new_path.back() ==
                pathAnalysis[pathAnalysis.size() - new_path.size()]) {
              pathsInfo_current.addPathValue(
                  new_path,
                  prev_path_score + score_before_merge - current_node->score,
                  current_node);
            }
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto& current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 ||
              (new_path.back() != current_node->tokNet && t != 0)) {
            new_path.push_back(current_node->tokNet);
          }
          if (new_path.back() ==
              pathAnalysis[pathAnalysis.size() - new_path.size()]) {
            pathsInfo_current.addPathValue(
                new_path,
                prev_path_score + score_no_merge - current_node->score,
                current_node);
          }
        }
      }
    }
    // pathsInfo_current.beamIt(beam_size);
  }

  double result = 0;
  for (auto const& path_tuple : pathsInfo_current.pathsInfo) {
    auto path = std::get<0>(path_tuple);
    std::reverse(path.begin(), path.end());
    if (path == pathAnalysis) {
      result = std::get<1>(path_tuple);
    }
  }
  return result;
}

std::tuple<
    double,
    std::vector<std::vector<std::tuple<std::vector<int>, double>>>>
ForceAlignBeamNoise::wLER(
    const fl::Variable& output,
    fl::Variable& cleantarget,
    int beam_size,
    fl::AverageValueMeter* mtr_wLER) {
  auto payload = output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data =
      std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)
          ->data->batch;

  int B = data.size();
  std::vector<double> all_wLER(B, 0);
  std::vector<std::vector<std::tuple<std::vector<int>, double>>>
      all_paths_weights(B);
#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    all_paths_weights[b] = extractPathsAndWeightsBackward(output, b, beam_size);
  }

  for (int b = 0; b < B; b++) {
    auto& paths_weights = all_paths_weights[b];
    fl::EditDistanceMeter mtr_LER;
    std::vector<double> probas_v;

    for (const auto& p_v : paths_weights) {
      probas_v.push_back(std::get<1>(p_v));
    }
    auto probas = fl::softmax(
        fl::Variable(af::array(probas_v.size(), probas_v.data()), false), 0);
    probas.host(probas_v.data());

    auto tgt_clean = cleantarget.array()(af::span, b);
    auto tgtraw_clean = w2l::afToVector<int>(tgt_clean);
    auto tgtsz_clean =
        w2l::getTargetSize(tgtraw_clean.data(), tgtraw_clean.size());
    tgtraw_clean.resize(tgtsz_clean);

    for (long j = 0; j < tgtraw_clean.size(); j++) {
      if (tgtraw_clean[j] == 28) {
        tgtraw_clean[j] = tgtraw_clean[j - 1];
      }
    }

    double wLER = 0.0;

    int idx = 0;
    for (const auto& p_v : paths_weights) {
      mtr_LER.reset();
      std::vector<int> path = std::get<0>(p_v);

      for (long j = 0; j < path.size(); j++) {
        if (path[j] == 28) {
          path[j] = path[j - 1];
        }
      }

      mtr_LER.add(
          path.data(), tgtraw_clean.data(), path.size(), tgtraw_clean.size());
      wLER += mtr_LER.value()[0] * probas_v[idx];
      idx++;
    }
    if (mtr_wLER != nullptr) {
      mtr_wLER->add(wLER);
    }
    all_wLER[b] = wLER;
  }

  double result = 0;
  for (auto& wLER : all_wLER) {
    result += wLER;
  }
  result /= (double)B;

  return std::make_tuple(result, all_paths_weights);
}
