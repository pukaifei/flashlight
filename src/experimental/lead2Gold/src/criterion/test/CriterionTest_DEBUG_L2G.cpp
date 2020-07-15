/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <array>

#include <criterion/criterion.h>
#include <iomanip>
#include "libraries/common/Dictionary.h"
#include "runtime/Serial.h"

#include <fstream>
#include <regex>
#include <string>

using namespace fl;
using namespace w2l;

namespace {

constexpr float kEpsilon = 1E-5;

void checkZero(const af::array& val, float precision = kEpsilon) {
  ASSERT_LE(af::max<float>(af::abs(val)), precision);
}

std::vector<float> readFloatFile(std::string filename, int size) {
  std::regex re("[+-]?([0-9]*[.])?[0-9]+"); // regex for a float//double//int
  std::ifstream f(filename);
  std::string line;

  if (!f.good()) {
    throw std::invalid_argument("could not read file");
  }
  float value;
  std::vector<float> res;

  while (std::getline(f, line)) { // for every line
    std::sregex_iterator next(line.begin(), line.end(), re);
    std::sregex_iterator end;
    while (next != end) { // for every column
      std::smatch match = *next;
      value = std::stof(match.str());
      res.push_back(value);
      next++;
    }
  }

  if (res.size() != size) {
    std::ostringstream error;
    error << "invalide size, got: " << res.size() << " expected: " << size;
    throw std::invalid_argument(error.str());
  }
}

std::vector<int> readIntFile(std::string filename, int size) {
  std::regex re("[+-]?([0-9]*[.])?[0-9]+"); // regex for a float//double//int
  std::ifstream f(filename);
  std::string line;

  if (!f.good()) {
    throw std::invalid_argument("could not read file");
  }
  int value;
  std::vector<int> res;

  while (std::getline(f, line)) { // for every line
    std::sregex_iterator next(line.begin(), line.end(), re);
    std::sregex_iterator end;
    while (next != end) { // for every column
      std::smatch match = *next;
      value = std::stoi(match.str());
      res.push_back(value);
      next++;
    }
  }

  if (res.size() != size) {
    std::ostringstream error;
    error << "invalide size, got: " << res.size() << " expected: " << size;
    throw std::invalid_argument(error.str());
  }
}

} // namespace

TEST(CriterionTestBEAM_DEBUG, ins_del_debug) {
  // const int N = 29, L = 190, T = 587, B = 3; // test with 28 + 1 rep label
  std::string savePath =
      "/checkpoint/adufraux/saveExample/80_M20_all_f1_no_ins_no_del.bin";

  af::array emission_af, transition_af, ltrTarget_af, ltrKeyTarget_af,
      ltrKeyTargetClean_af;

  W2lSerializer::load(
      savePath,
      emission_af,
      transition_af,
      ltrTarget_af,
      ltrKeyTarget_af,
      ltrKeyTargetClean_af);

  // af::print("emission_af", emission_af(af::span,af::seq(10,30),0));
  // af::print("transition_af", transition_af);
  // af::print("ltrTarget_af", ltrTarget_af);
  // af::print("ltrKeyTarget_af", ltrKeyTarget_af);
  // af::print("ltrKeyTargetClean_af", ltrKeyTargetClean_af);

  auto emissions = Variable(emission_af, true);
  auto target = Variable(ltrTarget_af, false); // with rep label
  auto ktarget = Variable(ltrKeyTarget_af, false);
  auto ktargetClean = Variable(ltrKeyTargetClean_af, false);
  // auto ktarget = Variable(af::array(L, B, ktarget_raw.data()), false);
  auto transitions = Variable(transition_af, true);

  int L = ktarget.dims(0);
  int N = emissions.dims(0);
  int T = emissions.dims(1);
  int B = emissions.dims(2);

  std::cout << "N: " << std::to_string(N) << " L: " << std::to_string(L)
            << " T: " << std::to_string(T) << " B: " << std::to_string(B)
            << std::endl;

  w2l::Dictionary noise_keys;
  std::shared_ptr<NoiseTrie> lex =
      std::shared_ptr<NoiseTrie>(new NoiseTrie(N, -1, nullptr));
  std::string token_list = "|'abcdefghijklmnopqrstuvwxyz";
  for (int i = 0; i < N - 1; i++) {
    lex->insert({i}, 0);
    std::string s(1, token_list[i]);
    noise_keys.addEntry(s, i);
  }
  // lex->insert({N-1}, 0);
  // std::shared_ptr<NoiseLMLetterSwapUnit> noiselm =
  // std::make_shared<NoiseLMLetterSwapUnit>("", "identitynoiselm", noise_keys,
  // true, false, false, false, 1.0);

  float scale_noise = 0.3;
  float scale_sub = 1;
  float scale_ins = 0.1;
  float scale_del = 0.1;
  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = std::make_shared<
      NoiseLMLetterSwapUnit>(
      "/checkpoint/adufraux/data/librispeech/chunks/noisy_datasets/from_clean100/probas/",
      "M20_f1_no_ins_no_del",
      noise_keys,
      true,
      false,
      false,
      false,
      scale_noise,
      scale_sub,
      scale_ins,
      scale_del);
  long beamsize = 10;
  double threshold = 0;

  auto fac_beam =
      ForceAlignBeamNoiseStats(noise_keys, lex, *noiselm, beamsize, threshold);
  auto fac_beam_output =
      fac_beam.forward(emissions, transitions, target, ktarget);
  // reverse to find which paths were actually agregate.

  // std::vector<float> fal_beam_output_host(B);
  // fal_beam_output.host(fal_beam_output_host.data());

  // for (int i = 0; i < B; i++) {
  //  std::cout << "beam: " <<  std::setprecision (15) <<
  //  fal_beam_output_host[i] << std::endl;
  // ASSERT_NEAR(fal_beam_output_host[i], -2.728, 1e-3);  // - 2.728 calculated
  // by hand
  //}

  ///////

  std::vector<float> fac_beam_output_host(B);
  fac_beam_output.host(fac_beam_output_host.data());

  auto payload = fac_beam_output.getPayload();
  if (!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }

  auto& data =
      std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(
          payload)
          ->data;

  int b = 0;
  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  std::cout << "noisy target: " << std::endl;
  af::print("ltrKeyTarget_af", af::transpose(ltrKeyTarget_af));

  std::cout << "final aggregation " << std::endl;
  ForceAlignBeamNoiseNodeStats aggreg_node = aggregateStatsNodes(hyps.at(T));

  std::cout << "MAX path " << std::endl;
  aggreg_node.displayStats("max");
  aggreg_node.displayMatrix("max", false, true, &noise_keys);
  aggreg_node.displayMatrix("max", true, true, &noise_keys);

  std::cout << "All paths weighted " << std::endl;
  aggreg_node.displayStats("weighted");
  aggreg_node.displayMatrix("weighted", false, true, &noise_keys);
  aggreg_node.displayMatrix("weighted", true, true, &noise_keys);

  for (int t = 1; t <= T; t++) {
    if (((t - 1) % 20 == 0) || t == T) {
      aggreg_node = aggregateStatsNodes(hyps.at(t));
      aggreg_node.displayStats("weighted");
    }
  }

  // fini.displayMatrix("max",true, true);
  // fini.displayMatrix("max",false, true);
  // fini.displayMatrix("weighted",true, true);
  // fini.displayMatrix("weighted",false, true);

  // for (int t=1; t <= T+1; t++){
  // auto& node = hyps.at(t).at(0);
  //}
  // std::cout << std::endl;

  // std::cout << "nb ins: ";
  // for (int t=1; t <= T+1; t++){
  //  auto& node = hyps.at(t).at(0);
  //  std::cout << node.nb_ins << ",";
  //}
  // std::cout << std::endl;

  // std::cout << "nb del: ";
  // for (int t=1; t <= T+1; t++){
  //  auto& node = hyps.at(t).at(0);
  //  std::cout << node.nb_del << ",";
  //}
  // std::cout << std::endl;

  // af::print("ltrKeyTargetClean_af", af::transpose(ltrKeyTargetClean_af));

  // auto fac_dynamic = ForceAlignmentCriterion(N, CriterionScaleMode::NONE);
  // fac_dynamic.setParams(transitions, 0);
  // auto fac_dynamic_output = fac_dynamic.forward(emissions, target);
  //

  // std::vector<float> fac_dynamic_output_host(B);
  // fac_dynamic_output.host(fac_dynamic_output_host.data());

  // for (int i = 0; i < B; i++) {
  // std::cout << "beam: " << fac_beam_output_host[i] << " dynamic: " <<
  // fac_dynamic_output_host[i] << std::endl;
  //  ASSERT_NEAR(fac_beam_output_host[i], fac_dynamic_output_host[i], 1e-4);
  //  //corresponds to 0.1% of the total probability
  //}

  // emissions.zeroGrad();
  // transitions.zeroGrad();
  // fac_beam_output.backward();
  // auto fac_beam_grad = emissions.grad().array();
  // auto fac_beam_trans_grad = transitions.grad().array();
  // af::print("fac_beam_grad", fac_beam_grad(af::span,af::seq(445,465),0),7);
  // af::print("fac_beam_trans_grad", fac_beam_trans_grad(af::span,af::span));

  // emissions.zeroGrad();
  // transitions.zeroGrad();
  // fac_dynamic_output.backward();
  // auto fac_dynamic_grad = emissions.grad().array();
  // auto fac_dynamic_trans_grad = transitions.grad().array();

  // af::print("fac_dynamic_grad",
  // fac_dynamic_grad(af::span,af::seq(445,465),0),7);
  // af::print("fac_dynamic_trans_grad",
  // fac_dynamic_trans_grad(af::span,af::span));

  // checkZero(fac_beam_grad - fac_dynamic_grad, 1e-5);
  // checkZero(fac_beam_trans_grad - fac_dynamic_trans_grad, 1e-4); // this one
  // is very sensitive to beamsize
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  //::testing::GTEST_FLAG(filter) =
  //"*fac_beam_type0Noise_as_fac_large_input_good_emission_prob*";
  //::testing::GTEST_FLAG(filter) = "*fac_beam_ins_del_as_fac_small_input*";
  ::testing::GTEST_FLAG(filter) = "CriterionTestBEAM_DEBUG*";
  return RUN_ALL_TESTS();
}
