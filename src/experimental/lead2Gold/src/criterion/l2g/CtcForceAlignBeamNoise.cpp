#include "experimental/lead2Gold/src/criterion/l2g/CtcForceAlignBeamNoise.h"

// for w2l::gettargetSize()
#include <assert.h>
#include <omp.h>
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

CtcForceAlignBeamNoise::CtcForceAlignBeamNoise(
    w2l::Dictionary& tokenDict,
    std::shared_ptr<NoiseTrie> lex,
    NoiseLMLetterSwapUnit& noiselm,
    long B,
    double threshold,
    int top_k,
    bool count_noise,
    bool count_noise_sort,
    int Nb_nested)
    : tokenDict_(tokenDict),
      lex_(lex),
      noiselm_(noiselm),
      B_(B),
      threshold_(threshold),
      top_k_(top_k),
      count_noise_(count_noise),
      count_noise_sort_(count_noise_sort),
      Nb_nested_(Nb_nested) {
  if (Nb_nested > 1) {
    omp_set_nested(true);
  } else {
    omp_set_nested(false);
  }
}

struct Comp { // to sort idx for top k feature
  Comp(const float* v) : _v(v) {}
  bool operator()(int a, int b) {
    return _v[a] > _v[b];
  }
  const float* _v;
};

fl::Variable CtcForceAlignBeamNoise::forward(
    fl::Variable& emissions,
    fl::Variable& noisytarget) {
  return CtcForceAlignBeamNoise::forward(emissions, emissions, noisytarget);
}

fl::Variable CtcForceAlignBeamNoise::forward(
    fl::Variable& emissions,
    fl::Variable& emissions_forsort,
    fl::Variable& noisytarget) {
  emissions = fl::logSoftmax(emissions, 0); // CTC style locally normalized
  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  // const int T = 5;
  const int B = emissions.dims(2);
  // const int B = 1;
  const int mS = noisytarget.dims(0);

  int top_k = top_k_ <= 0
      ? N
      : std::min(std::max(2, top_k_), N); // we don't allow top_k = 1

  std::vector<float> emissions_v(emissions.elements());
  std::vector<float> emissions_forsort_v(emissions_forsort.elements());
  std::vector<int> noisytarget_v(noisytarget.elements());
  emissions.host(emissions_v.data());
  emissions_forsort.host(emissions_forsort_v.data());
  noisytarget.host(noisytarget_v.data());

  auto data = std::make_shared<CtcForceAlignBeamNoiseBatchData>();
  data->batch.resize(B);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    auto emissions_p = emissions_v.data() + b * N * T;
    auto emissions_forsort_p = emissions_forsort_v.data() + b * N * T;
    auto noisytarget_p = noisytarget_v.data() + b * mS;
    const int S = w2l::getTargetSize(noisytarget_p, mS);
    // const int S=3;
    // noisytarget_p[0] = 0;
    // noisytarget_p[1] = 1;
    // noisytarget_p[2] = 2;

    // for  (int j=0 ; j < S ; j++){
    //  std::cout << tokenDict_.getEntry(noisytarget_p[j]);
    //}
    // std::cout << std::endl;

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    // +root +end
    hyps.resize(T + 2);
    std::vector<double> maxscores(T), maxscores_forsort(T);

    NoiseTrieNode* lexroot;
    if (lex_) {
      lexroot = lex_->root(); // start with the token "|" as root
    }

    hyps.at(0).emplace_back(
        tokenDict_.getIndex("|"), 0, 0, 0, -1, nullptr, lexroot, -1);

    double noisescore;
    double base_logProbNoise;
    double logProbNoise;
    double score, score_forsort;
    double baseScore, baseScore_forsort;
    long noiselmstate;
    int nb_remaining_target;
    int nb_remaining_frame;
    NoiseTrieNode* lexNode = nullptr; // useless ?

    long tok;
    std::vector<CtcForceAlignBeamNoiseNode> newhyps;

    std::vector<int> idx_unsort(N), idx_sorted(N);
    std::iota(idx_unsort.begin(), idx_unsort.end(), 0);
    idx_sorted = idx_unsort;
    int K;

    for (long t = 0; t < T; t++) {
      // std::cout << t << " / " << T << std::endl;
      newhyps.clear();
      if (!lex_ && top_k < N) { // prune top k
        idx_sorted = idx_unsort;
        std::partial_sort(
            idx_sorted.begin(),
            idx_sorted.begin() + top_k,
            idx_sorted.end(),
            Comp(emissions_forsort_p + t * N));
      }

      // std::vector<std::vector<CtcForceAlignBeamNoiseNode>>
      // newhypsNested(hyps.at(t).size()); #pragma omp parallel for
      //num_threads(Nb_nested_)
      for (auto& prev : hyps.at(t)) {
        // for(int prev_idx = 0;  prev_idx < hyps[t].size() ; prev_idx++) {
        // CtcForceAlignBeamNoiseNode& prev = hyps[t][prev_idx];
        // std::cout << "prev-> " << tokenDict_.getEntry(prev.tok) << " cursor:
        // "<< tokenDict_.getEntry(noisytarget_p[prev.noisytarget_t]) <<
        // std::endl; std::vector<CtcForceAlignBeamNoiseNode>& newhypsFromPrev =
        // newhypsNested[prev_idx]; add previous token if not in top k
        if (lex_) {
          idx_sorted.clear();
          for (auto child : prev.lexNode->children()) {
            if (child) {
              idx_sorted.push_back(child->idx());
            }
          }
          // add previous token if not already present
          if (std::find(idx_sorted.begin(), idx_sorted.end(), prev.tok) ==
              idx_sorted.end()) {
            idx_sorted.push_back(prev.tok);
          }
          // Add space token if lexNode is a leaf.
          if (!prev.lexNode->labels().empty()) {
            idx_sorted.push_back(tokenDict_.getIndex("|"));
          }
          // add blank token if not already present
          if (std::find(
                  idx_sorted.begin(),
                  idx_sorted.end(),
                  tokenDict_.getIndex(w2l::kBlankToken)) == idx_sorted.end()) {
            idx_sorted.push_back(tokenDict_.getIndex(w2l::kBlankToken));
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
          if (top_k < N && prev.tok >= 0 &&
              (std::find(
                   idx_sorted.begin(), idx_sorted.begin() + top_k, prev.tok) ==
               idx_sorted.begin() + top_k)) {
            idx_sorted[K] = prev.tok;
            K++;
          }
        }

        for (int idx = 0; idx < K; idx++) {
          int& tok = idx_sorted[idx];
          // std::cout << tok << " <- " << tokenDict_.getEntry(tok) <<
          // std::endl; std::cout <<
          // tokenDict_.getEntry(noisytarget_p[prev.noisytarget_t]) << " " <<
          // tokenDict_.getEntry(noisytarget_p[prev.noisytarget_t + 1]) << " " <<
          // tokenDict_.getEntry(noisytarget_p[prev.noisytarget_t + 2]) <<
          // std::endl;

          // tok = (i == tokenDict_.getIndex("1")) ? prev.tok : i;
          // tok = i; // Warning tok can be the blank label.

          if (lex_) { // Warning tok can be the blank label.
            if (tok == tokenDict_.getIndex(w2l::kBlankToken)) {
              lexNode = prev.lexNode;
            } else {
              lexNode = (tok == tokenDict_.getIndex("|"))
                  ? lexroot
                  : prev.lexNode->child(tok);
            }
          }

          if (t == 0) { // generate the first hypothesis except the rep label
            if (tok == tokenDict_.getIndex(w2l::kBlankToken) ||
                std::isfinite(noiselm_.scoreSwap(
                    noisytarget_p[prev.noisytarget_t + 1],
                    tok)) // If the next letter can be swapped
                || std::isfinite(noiselm_.scoreDeletion(
                       tok)) // if the next letter can be deleted
                ||
                (std::isfinite(noiselm_.scoreInsertion(
                     noisytarget_p[prev.noisytarget_t + 1])) &&
                 (prev.noisytarget_t + 2 <
                  S) // if the next letter can be inserted
                 && std::isfinite(noiselm_.scoreSwap(
                        noisytarget_p[prev.noisytarget_t + 2],
                        tok)) // and then the next one can be swapped or deleted
                              // (useless to verify since already done)
                 )) {
              baseScore = prev.score + emissions_p[t * N + tok];
              baseScore_forsort =
                  prev.score_forsort + emissions_forsort_p[t * N + tok];
              newhyps.emplace_back(
                  tok,
                  baseScore,
                  baseScore_forsort,
                  0,
                  prev.noisytarget_t + 1,
                  &prev,
                  lexNode,
                  noiselmstate);
              // std::cout << "New hyp " << "<-- " << tokenDict_.getEntry(tok)
              // << std::endl;
            }
          } else {
            baseScore = prev.score + emissions_p[t * N + tok];
            baseScore_forsort =
                prev.score_forsort + emissions_forsort_p[t * N + tok];

            nb_remaining_target = S - (prev.noisytarget_t + 1);
            nb_remaining_frame = T - (t + 1);

            if (prev.tok == tokenDict_.getIndex(w2l::kBlankToken)) {
              if ((tok == tokenDict_.getIndex(w2l::kBlankToken)) &&
                  (nb_remaining_frame >= nb_remaining_target + 1)) {
                newhyps.emplace_back(
                    tok,
                    baseScore,
                    baseScore_forsort,
                    0,
                    prev.noisytarget_t,
                    &prev,
                    lexNode,
                    noiselmstate);
                // std::cout << "New hyp " << "<-- " << tokenDict_.getEntry(tok)
                // << std::endl;
              }
              if ((tok != tokenDict_.getIndex(w2l::kBlankToken)) &&
                      (std::isfinite(noiselm_.scoreSwap(
                          noisytarget_p[prev.noisytarget_t],
                          tok))) // If the next letter can be swapped
                  ||
                  (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                   (nb_remaining_frame - 1 >=
                    nb_remaining_target)) // if the next letter can be deleted
                  ||
                  (std::isfinite(noiselm_.scoreInsertion(
                       noisytarget_p[prev.noisytarget_t])) &&
                   (prev.noisytarget_t + 2 <
                    S) // if the next letter can be inserted
                   &&
                   std::isfinite(noiselm_.scoreSwap(
                       noisytarget_p[prev.noisytarget_t + 1],
                       tok)) // and then the next one can be swapped or deleted
                             // (useless to verify since already done)
                   )) {
                newhyps.emplace_back(
                    tok,
                    baseScore,
                    baseScore_forsort,
                    0,
                    prev.noisytarget_t,
                    &prev,
                    lexNode,
                    noiselmstate);
                // std::cout << "New hyp " << "<-- " << tokenDict_.getEntry(tok)
                // << std::endl;
              }

            } else if (prev.tok == tok) {
              // we force to follow the target if we don't have enough frames to
              // finish it. Except for 1 letter if we allow insertion.
              if (nb_remaining_frame >= nb_remaining_target) {
                if (nb_remaining_frame >= nb_remaining_target + 1 ||
                    (std::isfinite(noiselm_.scoreSwap(
                        noisytarget_p[prev.noisytarget_t],
                        tok))) // If the next letter can be swapped
                    ||
                    (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                     (nb_remaining_frame - 1 >=
                      nb_remaining_target)) // if the next letter can be deleted
                    ||
                    (std::isfinite(noiselm_.scoreInsertion(
                         noisytarget_p[prev.noisytarget_t])) &&
                     (prev.noisytarget_t + 2 <
                      S) // if the next letter can be inserted
                     &&
                     std::isfinite(noiselm_.scoreSwap(
                         noisytarget_p[prev.noisytarget_t + 1],
                         tok)) // and then the next one can be swapped or
                               // deleted (useless to verify since already done)
                     )) {
                  newhyps.emplace_back(
                      tok,
                      baseScore,
                      baseScore_forsort,
                      0,
                      prev.noisytarget_t,
                      &prev,
                      prev.lexNode,
                      noiselmstate);
                  // std::cout << "New hyp " << "<-- " <<
                  // tokenDict_.getEntry(tok) << std::endl;
                }
              }
            } else {
              // or we change the letter, hence the key
              if (noiselm_.allowInsertion() &&
                  prev.noisytarget_t + 2 <
                      S) { // a letter has been added to the noisy
                           // transcription. +1 only because we can
                if ((tok == tokenDict_.getIndex(w2l::kBlankToken) &&
                     nb_remaining_frame >= nb_remaining_target - 1) ||
                    (tok != tokenDict_.getIndex(w2l::kBlankToken) &&
                         (std::isfinite(noiselm_.scoreSwap(
                             noisytarget_p[prev.noisytarget_t + 2],
                             tok))) // If the next letter can be swapped
                     || (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                         (nb_remaining_frame >= nb_remaining_target -
                              1)) // if the next letter can be deleted
                     || (std::isfinite(noiselm_.scoreInsertion(
                             noisytarget_p[prev.noisytarget_t + 2])) &&
                         (prev.noisytarget_t + 4 <
                          S) // if the next letter can be inserted
                         && std::isfinite(noiselm_.scoreSwap(
                                noisytarget_p[prev.noisytarget_t + 3],
                                tok)) // and then the next one can be swapped or
                                      // deleted (useless to verify since
                                      // already done)
                         ))) {
                  logProbNoise = noiselm_.scoreInsertion(
                                     noisytarget_p[prev.noisytarget_t]) +
                      noiselm_.scoreSwap(
                          noisytarget_p[prev.noisytarget_t + 1], prev.tok);
                  noisescore = noiselm_.scale_noise() * logProbNoise +
                      noiselm_.tkn_score();
                  if (std::isfinite(noisescore)) {
                    score = count_noise_ ? baseScore + noisescore : baseScore;
                    score_forsort = count_noise_sort_
                        ? baseScore_forsort + noisescore
                        : baseScore_forsort;
                    newhyps.emplace_back(
                        tok,
                        score,
                        score_forsort,
                        logProbNoise,
                        prev.noisytarget_t + 2,
                        &prev,
                        lexNode,
                        noiselmstate);
                    // std::cout << "New hyp " << "<-- " <<
                    // tokenDict_.getEntry(tok) << std::endl;
                  }
                }
              }

              base_logProbNoise = noiselm_.allowInsertion()
                  ? noiselm_.scoreNoInsertion()
                  : 0.; // no letter is inserted
              if (noiselm_.allowDeletion() &&
                  prev.noisytarget_t < S) { // for now allow only 1 deletion and
                                            // one last deletion
                // Verify that the proposed hyp is probable
                if ((tok == tokenDict_.getIndex(w2l::kBlankToken) &&
                     nb_remaining_frame >= nb_remaining_target + 1) ||
                    (tok != tokenDict_.getIndex(w2l::kBlankToken) &&
                         std::isfinite(noiselm_.scoreSwap(
                             noisytarget_p[prev.noisytarget_t],
                             tok)) // If the next letter can be swapped
                     || (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                         (nb_remaining_frame >= nb_remaining_target +
                              1)) // if the next letter can be deleted
                     || (std::isfinite(noiselm_.scoreInsertion(
                             noisytarget_p[prev.noisytarget_t])) &&
                         (prev.noisytarget_t + 2 <
                          S) // if the next letter can be inserted
                         && std::isfinite(noiselm_.scoreSwap(
                                noisytarget_p[prev.noisytarget_t + 1],
                                tok)) // and then the next one can be swapped or
                                      // deleted (useless to verify since
                                      // already done)
                         ))) {
                  // we allow a deletion only if we have enough frames to finish
                  // the sentence.
                  if (nb_remaining_frame >= nb_remaining_target) {
                    logProbNoise =
                        base_logProbNoise + noiselm_.scoreDeletion(prev.tok);
                    noisescore = noiselm_.scale_noise() * logProbNoise +
                        noiselm_.tkn_score();
                    // newhyps.push_back({i, baseScore + noisescore,
                    // prev.noisytarget_t, prev.noisytarget_t, &prev, letter,
                    // key, noiselmstate, prev.nb_subs, prev.nb_ins, prev.nb_del
                    // + 1});
                    if (std::isfinite(noisescore)) {
                      score = count_noise_ ? baseScore + noisescore : baseScore;
                      score_forsort = count_noise_sort_
                          ? baseScore_forsort + noisescore
                          : baseScore_forsort;
                      newhyps.emplace_back(
                          tok,
                          score,
                          score_forsort,
                          logProbNoise,
                          prev.noisytarget_t,
                          &prev,
                          lexNode,
                          noiselmstate);
                      // std::cout << "New hyp " << "<-- " <<
                      // tokenDict_.getEntry(tok) << std::endl;
                    }
                  }
                }
              }

              if (noiselm_.allowSwap() && prev.noisytarget_t + 1 < S) {
                // Verify that the proposed hyp is probable. Otherwise most of
                // the beam hyps are useless if the noise model is sparse.
                // std::cout << "swap " << tokenDict_.getEntry(tok)  << "remain
                // frame " << nb_remaining_frame << " remain tar " <<
                // nb_remaining_target << std::endl;
                if ((tok == tokenDict_.getIndex(w2l::kBlankToken) &&
                     nb_remaining_frame >= nb_remaining_target) ||
                    (tok != tokenDict_.getIndex(w2l::kBlankToken) &&
                         std::isfinite(noiselm_.scoreSwap(
                             noisytarget_p[prev.noisytarget_t + 1],
                             tok)) // If the next letter can be swapped
                     || (std::isfinite(noiselm_.scoreDeletion(tok)) &&
                         (nb_remaining_frame - 1 >= nb_remaining_target -
                              1)) // if the next letter can be deleted
                     || (std::isfinite(noiselm_.scoreInsertion(
                             noisytarget_p[prev.noisytarget_t + 1])) &&
                         (prev.noisytarget_t + 3 <
                          S) // if the next letter can be inserted
                         && std::isfinite(noiselm_.scoreSwap(
                                noisytarget_p[prev.noisytarget_t + 2],
                                tok)) // and then the next one can be swapped or
                                      // deleted (useless to verify since
                                      // already done)
                         ))) {
                  logProbNoise = base_logProbNoise +
                      noiselm_.scoreSwap(
                          noisytarget_p[prev.noisytarget_t], prev.tok);
                  noisescore = noiselm_.scale_noise() * logProbNoise +
                      noiselm_.tkn_score();
                  if (std::isfinite(noisescore)) {
                    score = count_noise_ ? baseScore + noisescore : baseScore;
                    score_forsort = count_noise_sort_
                        ? baseScore_forsort + noisescore
                        : baseScore_forsort;
                    newhyps.emplace_back(
                        tok,
                        score,
                        score_forsort,
                        logProbNoise,
                        prev.noisytarget_t + 1,
                        &prev,
                        lexNode,
                        noiselmstate);
                    // std::cout << "New hyp " << "<-- " <<
                    // tokenDict_.getEntry(tok) << std::endl;
                  }
                }
              }
            }
          }
        }
      }

      // flat vector of vector with move semantics.
      // for (const auto& sub : newhypsNested){
      //  std::move(sub.begin(), sub.end(), std::back_inserter(newhyps));
      //}
      // std::cout << "tot size hyps before merge " << newhyps.size() <<
      // std::endl;

      if (newhyps.size() == 0) {
        throw std::invalid_argument("No hypothesis generated");
      }

      // offset scores
      double maxscore = -std::numeric_limits<double>::infinity();
      double maxscore_forsort = -std::numeric_limits<double>::infinity();
      for (CtcForceAlignBeamNoiseNode& hyp : newhyps) {
        maxscore = std::max(maxscore, hyp.score);
        maxscore_forsort = std::max(maxscore_forsort, hyp.score_forsort);
      }
      for (CtcForceAlignBeamNoiseNode& hyp : newhyps) {
        hyp.score -= maxscore;
        hyp.score_forsort -= maxscore_forsort;
        hyp.maxscore = maxscore; // useless now ? pass pointeur ?
      }
      maxscores[t] = maxscore;
      maxscores_forsort[t] = maxscore_forsort;

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

      // merge identical nodes
      std::sort(
          newhyps.begin(),
          newhyps.end(),
          [](CtcForceAlignBeamNoiseNode& a, CtcForceAlignBeamNoiseNode& b) {
            if (a.lexNode == b.lexNode) {
              if (a.tok ==
                  b.tok) { // same as a.lex == b.lex but count the rep1 token
                if (a.noisytarget_t == b.noisytarget_t) {
                  return a.score_forsort > b.score_forsort;
                } else {
                  return a.noisytarget_t < b.noisytarget_t;
                }
              } else {
                return a.tok < b.tok;
              }
            } else {
              return a.lexNode < b.lexNode;
            }
          });

      long headidx = 0;
      long nhyp = newhyps.size();

      for (long h = 1; h < nhyp; h++) {
        CtcForceAlignBeamNoiseNode& elem = newhyps.at(h);
        CtcForceAlignBeamNoiseNode& head = newhyps.at(headidx);

        if ((head.lexNode == elem.lexNode) && (head.tok == elem.tok) &&
            (head.noisytarget_t ==
             elem.noisytarget_t)) { // maybe to change when key level
          if (head.merge_a < 0) {
            head.merge_a = merged.size();
            merged.push_back(head.parent); /* note: parent is in hyps */
            merged_score.push_back(head.score);
          }
          head.merge_b = merged.size();
          merged.push_back(elem.parent); /* note: parent is in hyps */
          merged_score.push_back(elem.score);
          head.score = logadd(head.score, elem.score);
          head.score_forsort = logadd(head.score_forsort, elem.score_forsort);
        } else {
          headidx++;
          if (headidx != h) {
            newhyps.at(headidx) = newhyps.at(h);
          }
        }
      }
      nhyp =
          std::min(headidx + 1, nhyp); // should be headidx+1 when it is
                                       // working. To handle the case nhyp = 0
      // beam it
      std::sort(
          newhyps.begin(),
          newhyps.begin() + nhyp,
          [](CtcForceAlignBeamNoiseNode& a, CtcForceAlignBeamNoiseNode& b) {
            return a.score_forsort > b.score_forsort;
          });
      nhyp = std::min(nhyp, B_);
      // std::cout << "tot size hyps after merge: " << nhyp << std::endl;
      hyps.at(t + 1).insert(
          hyps.at(t + 1).end(), newhyps.begin(), newhyps.begin() + nhyp);
    }
    CtcForceAlignBeamNoiseNode fini("fini");
    for (CtcForceAlignBeamNoiseNode& prev : hyps.at(T)) {
      if (fini.merge_a < 0) {
        fini.merge_a = merged.size();
        fini.merge_b = merged.size() - 1;
      }

      double score, score_forsort;
      double noisescore;
      double base_noisescore = 0.;
      bool merge_it = false;
      if (prev.tok == tokenDict_.getIndex(w2l::kBlankToken)) {
        merge_it = true;
        noisescore = 0;
      } else if (
          prev.noisytarget_t + 1 == S &&
          noiselm_.allowSwap()) { // no insertion + swap current + no insertion
                                  // after
        base_logProbNoise =
            noiselm_.allowInsertion() ? 2.0 * noiselm_.scoreNoInsertion() : 0.;
        logProbNoise = base_logProbNoise +
            noiselm_.scoreSwap(noisytarget_p[prev.noisytarget_t], prev.tok);
        noisescore =
            noiselm_.scale_noise() * logProbNoise + noiselm_.tkn_score();
        merge_it = true;
      } else if (
          prev.noisytarget_t + 2 == S &&
          noiselm_
              .allowInsertion()) { // insertion + swap current + no insertion
        base_logProbNoise = noiselm_.scoreNoInsertion() +
            noiselm_.scoreInsertion(noisytarget_p[prev.noisytarget_t]);
        logProbNoise = noiselm_.allowSwap() ? base_logProbNoise +
                noiselm_.scoreSwap(
                    noisytarget_p[prev.noisytarget_t + 1], prev.tok)
                                            : base_logProbNoise;
        noisescore = noiselm_.allowSwap()
            ? noiselm_.scale_noise() * logProbNoise + noiselm_.tkn_score()
            : noiselm_.scale_noise() * logProbNoise;
        merge_it = true;
      } else if (
          prev.noisytarget_t == S &&
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
        score = count_noise_ ? prev.score + noisescore : prev.score;
        score_forsort = count_noise_sort_ ? prev.score_forsort + noisescore
                                          : prev.score_forsort;

        fini.merge_b = merged.size();
        merged.push_back(&prev);
        merged_score.push_back(score);

        fini.score = logadd(fini.score, score);
        fini.score_forsort = logadd(fini.score_forsort, score_forsort);

        int n = fini.merge_b - fini.merge_a + 1;
      }
    }

    double sum_of_maxscores = 0.0;
    double sum_of_maxscores_forsort = 0.0;

    for (double& score : maxscores) {
      sum_of_maxscores += score;
    }
    for (double& score : maxscores_forsort) {
      sum_of_maxscores_forsort += score;
    }

    if (std::isfinite(sum_of_maxscores)) {
      fini.score += sum_of_maxscores;
    }
    if (std::isfinite(sum_of_maxscores_forsort)) {
      fini.score_forsort += sum_of_maxscores_forsort;
    }
    hyps.at(T + 1).push_back(std::move(fini));
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

  std::vector<fl::Variable> inputs = {emissions, noisytarget};

  return fl::Variable(
      result,
      inputs,
      grad_func,
      std::make_shared<CtcForceAlignBeamNoiseVariablePayload>(data));
}

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
    CtcForceAlignBeamNoiseNode& node,
    CtcForceAlignBeamNoiseNode& prev,
    float* gemissions_p,
    long t,
    long T,
    long N,
    NoiseLMLetterSwapUnit& noiselm,
    double g) {
  if (t >= 0 && t < T) {
    gemissions_p[t * N + node.tok] += g;
  }
  prev.gscore += g;
  prev.active = true;
}

void CtcForceAlignBeamNoise::backward(
    std::vector<fl::Variable>& inputs,
    const fl::Variable& goutput,
    std::shared_ptr<CtcForceAlignBeamNoiseBatchData> data) {
  auto& emissions = inputs[0];
  auto& noisytarget = inputs[1];

  const int N = emissions.dims(0);
  const int T = emissions.dims(1);
  const int B = emissions.dims(2);

  std::vector<float> gemissions_v(emissions.elements(), 0);
  std::vector<float> goutput_v(goutput.elements());
  goutput.host(goutput_v.data());

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    auto gemissions_p = gemissions_v.data() + b * N * T;
    double gscore = goutput_v[b];

    auto& hyps = data->batch[b].hyps;
    auto& merged = data->batch[b].merged;
    auto& merged_score = data->batch[b].merged_score;

    if (merged.size() != merged_score.size()) {
      std::cout << "$ merged scores have wrong sizes" << std::endl;
      throw std::invalid_argument("merged scores have wrong sizes");
    }

    for (std::vector<CtcForceAlignBeamNoiseNode>& hyps_t : hyps) {
      for (CtcForceAlignBeamNoiseNode& node : hyps_t) {
        node.gscore = 0;
        node.active = false;
      }
    }
    hyps.at(T + 1).at(0).active = true;
    hyps.at(T + 1).at(0).gscore = gscore;

    std::vector<double> sub_merged_score;
    std::vector<double> sub_merged_gscore;
    for (long t = T; t >= 0; t--) {
      for (CtcForceAlignBeamNoiseNode& node : hyps.at(t + 1)) {
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
  emissions.addGrad(
      fl::Variable(af::array(N, T, B, gemissions_v.data()), false));
}

af::array CtcForceAlignBeamNoise::viterbi(const fl::Variable& output) {
  auto payload = output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto data =
      std::dynamic_pointer_cast<CtcForceAlignBeamNoiseVariablePayload>(payload)
          ->data;
  int64_t B = data->batch.size();
  int64_t T = data->batch[0].hyps.size() - 2;

  std::vector<int> path_v(T * B);

  for (int64_t b = 0; b < B; b++) {
    auto path_p = path_v.data() + b * T;
    CtcForceAlignBeamNoiseNode* node = &(data->batch[b].hyps.at(T).at(0));
    int64_t t = T;
    while (node && (node->tok >= 0)) {
      path_p[--t] = node->tok;
      node = node->parent;
    }
    //    cleanViterbiPath(path_p, T);
  }
  return af::array(T, B, path_v.data());
}

/*
// implement threshold, resolve memory issues, max_nb_paths param
// don't sort every time, change the data structure

struct pathsInfo {
  std::map< std::vector<int>, double> pathsToValue; // paths
  std::multimap<double, decltype(pathsToValue.begin()) > reverseMap;
  int max_nb_paths = 50;
  double threshold = 1e-8;
  pathsInfo() {};
  pathsInfo(std::vector<int> path, double value) {
    pathsToValue[path] = value;
  };

  void addPathValue(std::vector<int> path, double value) {
    auto it_existing_path = pathsToValue.find(path);
    if (it_existing_path == pathsToValue.end()) { //if path is not present, we
may add it
      //if (value >= threshold){
        if (pathsToValue.size() < max_nb_paths){ // if we have enough space to
add a new path auto it = pathsToValue.insert({path, value}).first; // create a
new path and add value reverseMap.insert({value, it}); } else{ // else we have
to remove to worse one if the current is better auto it_min =
reverseMap.begin();

          if (it_min->first < value){ // if the lowest one is worse than the new
added value, we remove it and add the new. pathsToValue.erase(it_min->second);
// erase the corresponding iterator in pathsToValue reverseMap.erase(it_min); //
And erase it in the multimap. auto it = pathsToValue.insert({path,
value}).first; // create a new path and add value reverseMap.insert({value,
it});
          }
        }
    } else{
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

struct pathsInfoBeamBackward {
  typedef std::tuple<std::vector<int>, double,
std::map<ForceAlignBeamNoiseNode*, double>, double> pathTuple; typedef
std::tuple<std::vector<int>, double> simplePathTuple; std::vector<pathTuple>
pathsInfo;
  // This is a vector of path.
  // One path is represented by a tuple which contains:
  //  - get<0> : std::vector<int>, the actual path which is a vector a token.
  //  - get<1> : double, The associated score of this path.
  //  - get<2> : std::vector<ForceAlignBeamNoiseNodeStats*>, a vector containing
the adresses of the nodes that have lead to the path.
  //             We can have multiple nodes because of the merging operation
during the forward pass and because of the multiple alignements.


  pathsInfoBeamBackward() {};
  //pathsInfoBeam(std::vector<int> path, double value,
ForceAlignBeamNoiseNodeStats* node_ptr) {
  //  pathsInfo.emplace_back(std::make_tuple(path, value, node_ptr));
  //};

  void addPathValue(std::vector<int> path, double value,
ForceAlignBeamNoiseNode* node_ptr) {
    // Add the path if not present. Otherwise add the score of the already
stored path. auto it_existing_path = std::find_if(pathsInfo.begin(),
pathsInfo.end(),
                          [&path](const pathTuple& tuple) {return
(std::get<0>(tuple) == path);});
    // Verify if the path is present by comparing the first elemnents of the
tuples.

    if (it_existing_path == pathsInfo.end()) { //if the path is not present, we
add it std::map<ForceAlignBeamNoiseNode*, double> single_map = {{node_ptr,
value}}; pathTuple new_tuple = std::make_tuple(path, value, single_map,
node_ptr->score); pathsInfo.push_back(new_tuple); } else{
      std::get<1>(*it_existing_path) = logadd(std::get<1>(*it_existing_path),
value); // total score of the path currently

      auto& node_map = std::get<2>(*it_existing_path);
      auto it_exist_node = node_map.find(node_ptr);
      if (it_exist_node == node_map.end()){
        node_map[node_ptr] = value;
        //std::get<3>(*it_existing_path) =
logadd(std::get<3>(*it_existing_path), node_ptr->score);
        //std::get<3>(*it_existing_path) =
std::max(std::get<3>(*it_existing_path), node_ptr->score); } else{
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


    std::sort(pathsInfo.begin(), pathsInfo.end(), [](pathTuple& a, pathTuple& b)
{
        //return std::get<1>(a) + std::get<2>(a)->score > std::get<1>(b) +
std::get<2>(b)->score;
        //return std::get<1>(a) + std::get<3>(a) > std::get<1>(b) +
std::get<3>(b); return std::get<3>(a) > std::get<3>(b);
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



std::vector< std::tuple<std::vector<int>, double>>
ForceAlignBeamNoise::extractPathsAndWeightsBackward(const fl::Variable& output,
int b, int beam_size)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data =
std::dynamic_pointer_cast<ForceAlignBeamNoiseVariablePayload>(payload)->data->batch[b];

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
            auto& current_node = merged[idx + prev_node->merge_a]; // this tab
gives us directly the parent of the node

            score_before_merge = merged_score[idx + prev_node->merge_a] +
prev_node->maxscore;

            std::vector<int> new_path = prev_path;
            //std::cout << current_node << std::endl;
            if (new_path.size() == 0 || (new_path.back() != current_node->tokNet
&& t != 0)){ new_path.push_back(current_node->tokNet);
            }
            pathsInfo_current.addPathValue(new_path, prev_path_score +
score_before_merge - current_node->score, current_node);
          }
        } else {
          score_no_merge = prev_node->score + prev_node->maxscore;
          auto& current_node = prev_node->parent;

          std::vector<int> new_path = prev_path;
          if (new_path.size() == 0 || (new_path.back() != current_node->tokNet
&& t != 0)){ new_path.push_back(current_node->tokNet);
          }
          pathsInfo_current.addPathValue(new_path, prev_path_score +
score_no_merge - current_node->score, current_node);

        }

      }
    }
    pathsInfo_current.beamIt(beam_size);
  }
  return pathsInfo_current.getResult();
}



std::tuple<double, std::vector<std::vector< std::tuple<std::vector<int>,
double>>> > ForceAlignBeamNoise::wLER(const fl::Variable& output, fl::Variable&
cleantarget, int beam_size, fl::AverageValueMeter* mtr_wLER)
{
  auto payload = output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }
  auto& data =
std::dynamic_pointer_cast<CtcForceAlignBeamNoiseVariablePayload>(payload)->data->batch;

  int B = data.size();
  std::vector<double> all_wLER(B,0);
  std::vector<std::vector< std::tuple<std::vector<int>, double>>>
all_paths_weights(B); #pragma omp parallel for num_threads(B) for(int b = 0; b <
B; b++) { all_paths_weights[b] = extractPathsAndWeightsBackward(output, b,
beam_size);
  }

  for(int b = 0; b < B; b++) {
    auto& paths_weights = all_paths_weights[b];
    fl::EditDistanceMeter mtr_LER;
    std::vector<double> probas_v;

    for ( const auto &p_v : paths_weights ) {
      probas_v.push_back(std::get<1>(p_v));
    }
    auto probas = fl::softmax(fl::Variable(af::array(probas_v.size(),
probas_v.data()), false), 0); probas.host(probas_v.data());


    auto tgt_clean = cleantarget.array()(af::span, b);
    auto tgtraw_clean = w2l::afToVector<int>(tgt_clean);
    auto tgtsz_clean = w2l::getTargetSize(tgtraw_clean.data(),
tgtraw_clean.size()); tgtraw_clean.resize(tgtsz_clean);

    for (long j=0; j < tgtraw_clean.size(); j++){
      if (tgtraw_clean[j] == 28) {
        tgtraw_clean[j] = tgtraw_clean[j-1];
      }
    }

    double wLER=0.0;

    int idx = 0;
    for ( const auto &p_v : paths_weights ) {
      mtr_LER.reset();
      std::vector<int> path = std::get<0>(p_v);

      for (long j=0; j < path.size(); j++){
        if (path[j]==28){
          path[j] = path[j-1];
        }
      }

      mtr_LER.add(path.data(), tgtraw_clean.data(), path.size(),
tgtraw_clean.size()); wLER += mtr_LER.value()[0] * probas_v[idx]; idx++;
    }
    if (mtr_wLER != nullptr){
      mtr_wLER->add(wLER);
    }
    all_wLER[b] = wLER;
  }

  double result = 0;
  for (auto& wLER : all_wLER){
    result += wLER;
  }
  result /= (double)B;

return std::make_tuple(result, all_paths_weights);
}
*/
