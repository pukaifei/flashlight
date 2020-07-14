/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "experimental/lead2Gold/src/common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "experimental/lead2Gold/src/criterion/criterion.h"
#include "experimental/lead2Gold/src/data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include <omp.h>

using namespace w2l;
using namespace fl;

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

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " [data_path] [dataset_name] [flags]");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }


  /* ===================== Compute Stats ===================== */

  int N_=29;
  // test small
  //int N_=3;
  w2l::Dictionary noise_keys;
  std::shared_ptr<NoiseTrie> lex = std::shared_ptr<NoiseTrie>(new NoiseTrie(N_, -1, nullptr));
  std::string token_list = "|'abcdefghijklmnopqrstuvwxyz";
  for(int i = 0; i < N_-1; i++) {
    lex->insert({i}, 0);
    std::string s(1, token_list[i]);
    noise_keys.addEntry(s,i);
  }
  int nb_token = noise_keys.entrySize();

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
                                                    FLAGS_noiselmtype, noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
                                                    false, FLAGS_scale_noise, FLAGS_scale_sub, FLAGS_scale_ins, FLAGS_scale_del, FLAGS_tkn_score);

  std::cout << FLAGS_allowSwap << " " << FLAGS_allowInsertion << " " << FLAGS_allowDeletion << std::endl;
  // test small
  /*
  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
                                                    "zeronoiselm", noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
                                                      false, FLAGS_scale_noise, FLAGS_scale_ins, FLAGS_scale_del); 
  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm_display = noiselm;
  */

  std::shared_ptr<NoiseLMLetterSwapUnit> noiselm_display = std::make_shared<NoiseLMLetterSwapUnit>(FLAGS_probasdir,
                                                    FLAGS_noiselmtype, noise_keys, FLAGS_allowSwap, FLAGS_allowInsertion, FLAGS_allowDeletion,
                                                    false, 1, FLAGS_scale_sub, FLAGS_scale_ins, FLAGS_scale_del, FLAGS_tkn_score);


  int max_thread = omp_get_max_threads();
  std::cout << "MAX thread: " << max_thread << std::endl;

  auto fac_beam = ForceAlignBeamNoiseStats(noise_keys, lex, *noiselm, FLAGS_beamsize, FLAGS_beamthreshold, FLAGS_topk);
  auto fac_beam_normal = ForceAlignBeamNoise(noise_keys, lex, *noiselm, FLAGS_beamsize, FLAGS_beamthreshold, FLAGS_topk);




  //Get the paths of the files
  std::vector<std::string> paths;
  std::string line;
  std::ifstream myfile (FLAGS_saveExamplePathFolder + "/list_files.txt");
  while ( std::getline (myfile,line) )
    {
      paths.push_back(line);
    }
  myfile.close();


  int tot_B = paths.size();

  tot_B = FLAGS_nb_ex;
  int nthreads = FLAGS_nthread;

  std::vector<ForceAlignBeamNoiseNodeStats> aggregated_nodes(tot_B);
  std::vector<std::string> batchStrings(tot_B);

#pragma omp parallel for num_threads(nthreads)
  for(int b=0; b < tot_B; b++){
    std::string path = paths[b];

    af::array emission_af, transition_af, ltrTarget_af, ltrKeyTarget_af, ltrKeyTargetClean_af;
    fl::Variable emissions, target, ktarget, ktargetClean, transitions;
    int L, N, T, B;

    W2lSerializer::load(path, emission_af, transition_af, ltrTarget_af, ltrKeyTarget_af, ltrKeyTargetClean_af);

    emissions = Variable(emission_af, true);
    target = Variable(ltrTarget_af, false); // with rep label
    ktarget = Variable(ltrKeyTarget_af, false);
    ktargetClean = Variable(ltrKeyTargetClean_af, false);
    transitions = Variable(transition_af, true);





    std::vector<int> ktarget_token_v(ktarget.elements());
    ktarget.host(ktarget_token_v.data());
    auto tgtsz = w2l::getTargetSize(ktarget_token_v.data(), ktarget_token_v.size());
    ktarget_token_v.resize(tgtsz);

    std::vector<int> ktargetClean_token_v(ktargetClean.elements());
    ktargetClean.host(ktargetClean_token_v.data());
    auto tgtcleansz = w2l::getTargetSize(ktargetClean_token_v.data(), ktargetClean_token_v.size());
    ktargetClean_token_v.resize(tgtcleansz);



    L = ktarget.dims(0);
    N = emissions.dims(0);
    T = emissions.dims(1);
    B = emissions.dims(2);



    auto fac_beam_output = fac_beam.forward(emissions, transitions, target, ktarget);


    std::vector<float> score_beam_v(fac_beam_output.elements());
    fac_beam_output.host(score_beam_v.data());
    double score_beam = score_beam_v[0];

    auto fac_beam_output_normal = fac_beam_normal.forward(emissions, transitions, target, ktarget);


    //
    std::cout << "compute paths" << std::endl;

    std::vector< std::tuple<std::vector<int>, double>> paths_and_weights;
    std::vector<int> greedy_path;
    fl::TimeMeter meter;
    meter.resume();
    if (FLAGS_statsDirection == "backward"){
      std::cout << "Backward normal, B = " << FLAGS_beamsize << " statsbeam = " << FLAGS_statsbeamsize << std::endl;
      //paths_and_weights = fac_beam.extractPathsAndWeightsBackward(fac_beam_output, 0, FLAGS_statsbeamsize);
      paths_and_weights = fac_beam_normal.extractPathsAndWeightsBackward(fac_beam_output_normal, 0, FLAGS_statsbeamsize);

      greedy_path = std::get<0>(fac_beam.greedyPath(fac_beam_output, 0)[0]);
    } else if (FLAGS_statsDirection == "backward_simple") {
      std::cout << "Backward simplify, B = " << FLAGS_beamsize << " statsbeam = " << FLAGS_statsbeamsize << std::endl;
      paths_and_weights = fac_beam.extractPathsAndWeightsBackwardSimplify(fac_beam_output, 0, FLAGS_statsbeamsize);
      greedy_path = std::get<0>(fac_beam.greedyPath(fac_beam_output, 0)[0]);
    } else if (FLAGS_statsDirection == "greedy") {
      paths_and_weights = fac_beam.greedyPath(fac_beam_output, 0);
      greedy_path = std::get<0>(paths_and_weights[0]);
    } else{
      std::cout << "Both, B = " << FLAGS_beamsize << " statsbeam = " << FLAGS_statsbeamsize << std::endl;
      paths_and_weights = fac_beam.extractPathsAndWeightsBoth(fac_beam_output, 0, FLAGS_statsbeamsize);
      greedy_path = std::get<0>(fac_beam.greedyPath(fac_beam_output, 0)[0]);
    }
    meter.stop();
    

    std::cout << "done in " << meter.value() << std::endl;
    std::cout << "clean: ";
    for (int j=0; j<ktargetClean_token_v.size(); j++){
      if (ktargetClean_token_v[j] == 28){
        std::cout << noise_keys.getEntry(ktargetClean_token_v[j-1]);
      } else{
        std::cout << noise_keys.getEntry(ktargetClean_token_v[j]);
      }
    }
    std::cout << std::endl;

    std::cout << "provided: ";
    for (int j=0; j<ktarget_token_v.size(); j++){
      if (ktarget_token_v[j] == 28){
        std::cout << noise_keys.getEntry(ktarget_token_v[j-1]);
      } else{
        std::cout << noise_keys.getEntry(ktarget_token_v[j]);
      }
    }
    std::cout << std::endl;


    std::cout << "display path (" << paths_and_weights.size() << ")" << std::endl;


    double tot_score = -std::numeric_limits<double>::infinity();
    for (auto& path_weight : paths_and_weights){
      tot_score = logadd(tot_score, std::get<1>(path_weight));

      auto& proposed_path = std::get<0>(path_weight);
      //auto trueScore = fac_beam.getTrueScore(fac_beam_output, 0, proposed_path);

      std::cout << std::get<1>(path_weight) << ": ";
      for (int j=0; j<proposed_path.size(); j++){
        if (proposed_path[j] == 28){
          std::cout << noise_keys.getEntry(proposed_path[j-1]);
        } else{
          std::cout << noise_keys.getEntry(proposed_path[j]);
        }
      }
      if (proposed_path == greedy_path){
        std::cout << " <-- Greedy path";
      }
      //std::cout << " <-- trueScore : " << trueScore;
      std::cout << std::endl;
    }

     std::cout << "Greedy path: ";
    for (int j=0; j<greedy_path.size(); j++){
      if (greedy_path[j] == 28){
        std::cout << noise_keys.getEntry(greedy_path[j-1]);
      } else{
        std::cout << noise_keys.getEntry(greedy_path[j]);
      }
    }
    std::cout << std::endl;
    

    std::cout << "score beam " << score_beam << std::endl;
    af::print("fac_beam_output", fac_beam_output.array());
    std::cout << "tot score paths: " << tot_score << std::setprecision(15) << std::endl;

    auto payload = fac_beam_output.getPayload();
    auto& data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;
    auto& hyps = data->batch[0].hyps;

    if (!FLAGS_computeStatsStorePath.empty()){
      std::string toStore = "";
      std::vector<double> res_m, res_w;
      std::vector<double> count_sub_M(T), count_ins_M(T), count_del_M(T), length_M(T), target_used_M(T);
      std::vector<double> count_sub_W(T), count_ins_W(T), count_del_W(T), length_W(T), target_used_W(T);
      std::vector<std::vector<std::vector<double>>> sub_token_M(T), sub_token_W(T);
      std::vector<std::vector<double>> del_token_M(T), del_token_W(T);
      std::vector<std::vector<double>> ins_token_M(T), ins_token_W(T);

      for (int t = 1; t <= T; t++) {
        ForceAlignBeamNoiseNodeStats aggreg_node_t = aggregateStatsNodes(hyps.at(t));
        //Get edit stats results
        res_m = aggreg_node_t.getEditStats("max");
        count_sub_M[t-1] = res_m[0];
        count_ins_M[t-1] = res_m[1];
        count_del_M[t-1] = res_m[2];
        length_M[t-1] = res_m[3];
        target_used_M[t-1] = res_m[4];

        sub_token_M[t-1] = aggreg_node_t.sub_token;
        del_token_M[t-1] = aggreg_node_t.del_token;
        ins_token_M[t-1] = aggreg_node_t.ins_token;


        res_w = aggreg_node_t.getEditStats("weighted");
        count_sub_W[t-1] = res_w[0];
        count_ins_W[t-1] = res_w[1];
        count_del_W[t-1] = res_w[2];
        length_W[t-1] = res_w[3];
        target_used_W[t-1] = res_w[4];

        sub_token_W[t-1] = aggreg_node_t.w_sub_token;
        del_token_W[t-1] = aggreg_node_t.w_del_token;
        ins_token_W[t-1] = aggreg_node_t.w_ins_token;

        //
      }
      toStore += "{\n";


      toStore += "\"best_paths\": [";
      double tot_score_best_paths = -std::numeric_limits<double>::infinity();
      for (int i=0; i<paths_and_weights.size(); i++){
        auto& path_weight = paths_and_weights[i];
        auto& path = std::get<0>(path_weight);
        auto& weight = std::get<1>(path_weight);
        tot_score_best_paths = logadd(tot_score_best_paths, weight);

        toStore += "[" + std::to_string(weight) + ", [";
        for (int j=0; j<path.size(); j++){
          if (path[j] == 28){
            toStore += std::to_string(path[j-1]);
          } else{
            toStore += std::to_string(path[j]);
          }
          if (j != path.size() - 1){toStore +=", ";}
        }

        toStore += "], \"";

        for (int j=0; j<path.size(); j++){
          if (path[j] == 28){
            toStore += noise_keys.getEntry(path[j-1]);
          } else{
            toStore += noise_keys.getEntry(path[j]);
          }
        }

        toStore += "\"]";
        if (i != paths_and_weights.size() - 1){toStore +=", ";}
      }
      toStore += "],\n";


      if (tot_score_best_paths != -std::numeric_limits<double>::infinity()){
        toStore += "\"tot_score_best_paths\": " + std::to_string(tot_score_best_paths) + ",\n";
      }
      else{
        toStore += "\"tot_score_best_paths\": \"-inf\",\n";
      }

      toStore += "\"score_beam\": " + std::to_string(score_beam) + ",\n";



      toStore += "\"provided_target_int\": [";
      for (int i=0; i<ktarget_token_v.size(); i++){
        toStore += std::to_string(ktarget_token_v[i]) ;
        if (i != ktarget_token_v.size() - 1){toStore +=", ";}
      }
      toStore += "],\n";

      toStore += "\"provided_target\": \"";
      for (int i=0; i<ktarget_token_v.size(); i++){
        toStore += noise_keys.getEntry(ktarget_token_v[i]) ;
      }
      toStore += "\",\n";

      toStore += "\"clean_target_int\": [";
      for (int i=0; i<ktargetClean_token_v.size(); i++){
        if (ktargetClean_token_v[i] == 28){
          toStore += std::to_string(ktargetClean_token_v[i-1]);
        }else{
          toStore += std::to_string(ktargetClean_token_v[i]);
        }
        if (i != ktargetClean_token_v.size() - 1){toStore +=", ";}
      }
      toStore += "],\n";

      toStore += "\"clean_provided_target\": \"";
      for (int i=0; i<ktargetClean_token_v.size(); i++){
        if (ktargetClean_token_v[i] == 28){
          toStore += noise_keys.getEntry(ktargetClean_token_v[i-1]);
        }else{
          toStore += noise_keys.getEntry(ktargetClean_token_v[i]);
        }
      }
      toStore += "\",\n";


      toStore += "\"tot_target_size\": " + std::to_string(hyps.at(1).at(0).target_size);

      if (FLAGS_computeStatsLight){
        toStore += "\n";
      } else{
        toStore += ",\n";

        toStore += "\"count_sub_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_sub_M[t-1]) + ", ";
        }
        toStore += std::to_string(count_sub_M[T-1]);
        toStore += "],\n";

        toStore += "\"count_sub_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_sub_W[t-1]) + ", ";
        }
        toStore += std::to_string(count_sub_W[T-1]);
        toStore += "],\n";

        toStore += "\"count_ins_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_ins_M[t-1]) + ", ";
        }
        toStore += std::to_string(count_ins_M[T-1]);
        toStore += "],\n";

        toStore += "\"count_ins_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_ins_W[t-1]) + ", ";
        }
        toStore += std::to_string(count_ins_W[T-1]);
        toStore += "],\n";

        toStore += "\"count_del_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_del_M[t-1]) + ", ";
        }
        toStore += std::to_string(count_del_M[T-1]);
        toStore += "],\n";

        toStore += "\"count_del_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(count_del_W[t-1]) + ", ";
        }
        toStore += std::to_string(count_del_W[T-1]);
        toStore += "],\n";

        toStore += "\"length_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(length_M[t-1]) + ", ";
        }
        toStore += std::to_string(length_M[T-1]);
        toStore += "],\n";

        toStore += "\"length_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(length_W[t-1]) + ", ";
        }
        toStore += std::to_string(length_W[T-1]);
        toStore += "],\n";

        toStore += "\"target_used_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(target_used_M[t-1]) + ", ";
        }
        toStore += std::to_string(target_used_M[T-1]);
        toStore += "],\n";

        toStore += "\"target_used_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += std::to_string(target_used_W[t-1]) + ", ";
        }
        toStore += std::to_string(target_used_W[T-1]);
        toStore += "],\n";

        toStore += "\"sub_token_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += "[";
            for (int j = 0; j<nb_token; j++){
              toStore += std::to_string(sub_token_M[t-1][i][j]);
              if (j != nb_token-1){toStore += ", ";}
            }
            toStore += "]";
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "],\n";

        toStore += "\"sub_token_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += "[";
            for (int j = 0; j<nb_token; j++){
              toStore += std::to_string(sub_token_W[t-1][i][j]);
              if (j != nb_token-1){toStore += ", ";}
            }
            toStore += "]";
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "],\n";


        toStore += "\"ins_token_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += std::to_string(ins_token_M[t-1][i]);
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "],\n";


        toStore += "\"ins_token_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += std::to_string(ins_token_W[t-1][i]);
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "],\n";

        toStore += "\"del_token_max\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += std::to_string(del_token_M[t-1][i]);
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "],\n";

        toStore += "\"del_token_weighted\": [";
        for (int t = 1; t < T; t++) {
          toStore += "[";
          for (int i = 0; i<nb_token; i++){
            toStore += std::to_string(del_token_W[t-1][i]);
            if (i != nb_token-1){toStore += ", ";}
          }
          toStore += "]";
          if (t != T-1){toStore += ", ";}
        }
        toStore += "]\n";

      }


      toStore += "}";

      batchStrings[b] = toStore;
    }
    aggregated_nodes[b] = aggregateStatsNodes(hyps.at(T));
  }

  ForceAlignBeamNoiseNodeStats aggreg_node_tot = aggregated_nodes[0];
  for(int b=1; b < tot_B; b++){
    aggreg_node_tot.addStatsFromOtherNode(aggregated_nodes[b]);
  }

  //std::cout <<"MAX path " << std::endl;
  //aggreg_node_tot.displayStats("max");
  //aggreg_node_tot.displayMatrix("max", false, true, &noise_keys);
  //aggreg_node_tot.displayMatrix("max", true, true, &noise_keys);

  std::cout <<"All paths weighted " << std::endl;
  aggreg_node_tot.displayStats("weighted");
  aggreg_node_tot.displayMatrix("weighted", false, true, &noise_keys);
  aggreg_node_tot.displayMatrix("weighted", true, true, &noise_keys);

  displayNoiseModel(*noiselm_display, true, &noise_keys);


  if (!FLAGS_computeStatsStorePath.empty()){
    std::ofstream outfile (FLAGS_computeStatsStorePath);
    if (outfile.is_open()){
      outfile << "{\n";
      //output B nodes info
      for(int b=0; b < tot_B; b++){
        outfile << "\"" + std::to_string(b) + "\":" << batchStrings[b] << ",\n";
        //if (b != tot_B -1){outfile << ",";}
        //outfile << "\n";
      }

      outfile << "\"keys\": [";
      for (int i = 0; i < noise_keys.entrySize(); ++i)
      {
        outfile << "\"" << noise_keys.getEntry(i) << "\"";
        if (i != noise_keys.entrySize()-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"noise model sub\": [";
        for (int i = 0; i<nb_token; i++){
        outfile << "[";
        for (int j = 0; j<nb_token; j++){
          outfile << std::to_string(exp(noiselm_display->scoreSwap(i,j))*100);
          if (j != nb_token-1){outfile << ", ";}
        }
        outfile << "]";
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"noise model ins\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(exp(noiselm_display->scoreInsertion(i))*100);
        outfile << ", ";
      }
      outfile << std::to_string(exp(noiselm_display->scoreNoInsertion())*100);
      outfile << "],\n";

      outfile << "\"noise model del\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(exp(noiselm_display->scoreDeletion(i))*100);
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";


      //output aggregated_tot node info
      std::vector<double> res_m = aggreg_node_tot.getEditStats("max");
      std::vector<double> res_w = aggreg_node_tot.getEditStats("weighted");

      outfile << "\"total\": {\n";
      outfile << "\"count_sub_max\": " << std::to_string(res_m[0]) << ",\n";
      outfile << "\"count_sub_weighted\": " << std::to_string(res_w[0])<< ",\n";
      outfile << "\"count_ins_max\": " << std::to_string(res_m[1])<< ",\n";
      outfile << "\"count_ins_weighted\": " << std::to_string(res_w[1])<< ",\n";
      outfile << "\"count_del_max\": " << std::to_string(res_m[2])<< ",\n";
      outfile << "\"count_del_weighted\": " << std::to_string(res_w[2])<< ",\n";
      outfile << "\"length_max\": " << std::to_string(res_m[3])<< ",\n";
      outfile << "\"length_weighted\": " << std::to_string(res_w[3])<< ",\n";
      outfile << "\"target_used_max\": " << std::to_string(res_m[4])<< ",\n";
      outfile << "\"target_used_weighted\": " << std::to_string(res_w[4])<< ",\n";

      outfile << "\"sub_token_max\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << "[";
        for (int j = 0; j<nb_token; j++){
          outfile << std::to_string(aggreg_node_tot.sub_token[i][j]);
          if (j != nb_token-1){outfile << ", ";}
        }
        outfile << "]";
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"sub_token_weighted\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << "[";
        for (int j = 0; j<nb_token; j++){
          outfile << std::to_string(aggreg_node_tot.w_sub_token[i][j]);
          if (j != nb_token-1){outfile << ", ";}
        }
        outfile << "]";
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"ins_token_max\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(aggreg_node_tot.ins_token[i]);
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"ins_token_weighted\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(aggreg_node_tot.w_ins_token[i]);
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"del_token_max\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(aggreg_node_tot.del_token[i]);
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "],\n";

      outfile << "\"del_token_weighted\": [";
      for (int i = 0; i<nb_token; i++){
        outfile << std::to_string(aggreg_node_tot.w_del_token[i]);
        if (i != nb_token-1){outfile << ", ";}
      }
      outfile << "]\n";







      outfile << "}\n";

      outfile << "}";
      outfile.close();
    } else{
      std::cout << "Unable to open file " << FLAGS_computeStatsStorePath << std::endl;;
    }
  }


  //storeResults(hyps, storePath);
     

  //auto fac_beam_output = fac_beam.forward(emissions, transitions, target, ktarget);


  /*
  auto emi_b = emissions(af::span,af::span,0);
  auto tar_b = target(af::span,0);
  auto ktar_b = ktarget(af::span,0);
  auto fac_beam_output = fac_beam.forward(emi_b, transitions, tar_b, ktar_b);
  //auto fac_beam_output = fac_beam.forward(emissions, transitions, target, ktarget);
  std::vector<float> fac_beam_output_host(B);
  fac_beam_output.host(fac_beam_output_host.data());

  auto payload = fac_beam_output.getPayload();
  if(!payload) {
    throw std::invalid_argument("expecting a payload on provided Variable");
  }

  auto& data = std::dynamic_pointer_cast<ForceAlignBeamNoiseStatsVariablePayload>(payload)->data;

  int b=0;
  auto& hyps = data->batch[b].hyps;
  auto& merged = data->batch[b].merged;
  auto& merged_score = data->batch[b].merged_score;

  std::cout << "noisy target: " << std::endl;
  af::print("ltrKeyTarget_af", af::transpose(ltrKeyTarget_af));


  std::cout <<"final aggregation " << std::endl;
  ForceAlignBeamNoiseNodeStats aggreg_node = aggregateStatsNodes(hyps.at(T));

  std::cout <<"MAX path " << std::endl;
  aggreg_node.displayStats("max");
  aggreg_node.displayMatrix("max", false, true, &noise_keys);
  aggreg_node.displayMatrix("max", true, true, &noise_keys);

  std::cout <<"All paths weighted " << std::endl;
  aggreg_node.displayStats("weighted");
  aggreg_node.displayMatrix("weighted", false, true, &noise_keys);
  aggreg_node.displayMatrix("weighted", true, true, &noise_keys);

  for (int t = 1; t <= T; t++) {
    if (((t-1) % 20 == 0) || t==T){
      aggreg_node = aggregateStatsNodes(hyps.at(t));
      aggreg_node.displayStats("weighted");
    }
  }
  */
  

  return 0;
}
