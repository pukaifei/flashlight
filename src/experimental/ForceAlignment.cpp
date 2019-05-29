/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ForceAlignment.h"

namespace w2l {

ForceAlignment::ForceAlignment(const std::vector<float>& transition)
    : transraw_(transition) {}

std::vector<std::vector<int>> ForceAlignment::align(
    const fl::Variable& input,
    const fl::Variable& target) {
  int N = input.dims(0);
  int T = input.dims(1);
  int B = input.dims(2);
  int L = target.dims(0);

  if (N * N != transraw_.size()) {
    throw(af::exception("FAC: N doesn't match with the letter size"));
  }

  auto fal_buf = FalParameters(N, T, B, L);
  target.host(fal_buf.targetraw.data());
  input.host(fal_buf.inputraw.data());
  std::vector<std::vector<int>> bestPaths;

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    float* inp_p = fal_buf.inputraw.data() + b * N * T;
    double* falacc_p = fal_buf.falacc.data() + b * L * T;
    auto target_p = fal_buf.targetraw.data() + b * L;
    auto backptr_p = fal_buf.backptr.data() + b * L * T;

    int TN = w2l::getTargetSize(target_p, L);
    if (TN > T || TN == 0) {
      fal_buf.fal[b] = NAN;
      continue;
    }

    falacc_p[0] = inp_p[target_p[0]];
    backptr_p[0] = -1;

    double* s1i = fal_buf.s1i.data() + b * L;
    double* s2i = fal_buf.s2i.data() + b * L;

    for (int i = 0; i < TN; i++) {
      s1i[i] = transraw_[N * (target_p[i]) + target_p[i]];
      s2i[i] = i > 0 ? transraw_[N * (target_p[i]) + target_p[i - 1]] : 0;
    }
    for (int t = 1; t < T; t++) {
      double* falacc_t_prev = falacc_p + (t - 1) * TN;
      double* falacc_t = falacc_p + t * TN;
      int* backptr_t = backptr_p + t * TN;
      const float* inp_t = inp_p + t * N;
      int high = t < TN ? t : TN;
      int low = T - t < TN ? TN - (T - t) : 1;

      if (T - t >= TN) {
        falacc_t[0] = s1i[0] + falacc_t_prev[0] + inp_t[target_p[0]];
      }
      for (int i = low; i < high; i++) {
        double s1 = s1i[i] + falacc_t_prev[i];
        double s2 = s2i[i] + falacc_t_prev[i - 1];
        if (s1 > s2) {
          falacc_t[i] = s1 + inp_t[target_p[i]];
          backptr_t[i] = 1;
        } else {
          falacc_t[i] = s2 + inp_t[target_p[i]];
          backptr_t[i] = 2;
        }
      }
      if (high < TN) {
        falacc_t[high] =
            s2i[high] + falacc_t_prev[high - 1] + inp_t[target_p[high]];
      }
    }

    // Not used currently
    // Still keeping this, we could use it to filter
    // noisy transcripts if needed in future
    fal_buf.fal[b] = static_cast<float>(falacc_p[T * TN - 1]);

    // Compute the best path using backptrs
    auto ltrIdx = TN - 1;
    std::vector<int> bestPath(T);
    for (auto t = T - 1; t >= 0; t--) {
      bestPath[t] = target_p[ltrIdx];
      if (backptr_p[t * TN + ltrIdx] == 2) {
        ltrIdx--;
      }
    }
    bestPaths.emplace_back(bestPath);
  }
  return bestPaths;
}

} // namespace w2l
