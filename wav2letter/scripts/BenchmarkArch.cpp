/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include <flashlight/flashlight.h>
#include "common/FlashlightUtils.h"
#include "module/W2lModule.h"

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  af::sync();

  int num_iters = 100;
  af::sync();
  auto start = af::timer::start();
  for (int i = 0; i < num_iters; i++) {
    fn();
  }
  af::sync();
  return af::timer::stop(start) * 1000.0 / num_iters;
}

int main(int argc, char** argv) {
  af::info();
  if (argc < 4) {
    std::cout
        << "Invalid arguments. Usage : <binary> <archfile> <indim> <outdim>"
        << std::endl;
    return 1;
  }

  auto network =
      w2l::createW2lSeqModule(argv[1], std::stoi(argv[2]), std::stoi(argv[3]));

  std::cout << "[Network] arch - " << network->prettyString() << std::endl;
  std::cout << "[Network] params - " << w2l::numTotalParams(network)
            << std::endl;

  auto input =
      fl::Variable(af::randu(10 * 100, std::stoi(argv[2])), false); // 10 sec

  // forward
  network->eval();
  auto fwd_fn = [&]() {
    network->zeroGrad();
    input.zeroGrad();
    auto output = network->forward({input});
  };

  std::cout << "Network fwd took " << timeit(fwd_fn) << "ms" << std::endl;

  // fwd + bwd
  network->train();
  auto fwd_bwd_fn = [&]() {
    network->zeroGrad();
    input.zeroGrad();
    auto output = network->forward({input});
    output.backward();
  };

  std::cout << "Network fwd+bwd took " << timeit(fwd_bwd_fn) << "ms"
            << std::endl;

  return 0;
}
