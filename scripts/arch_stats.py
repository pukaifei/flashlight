"""
Script to print stats about wav2letter arch file.

Usage: arch_stats.py [-h] -a ARCHFILE

Arguments:
  -a ARCHFILE, --arch ARCHFILE Path to W2L architecture file
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W2L Architecture Stats")
    parser.add_argument(
        "-a", "--arch", type=str, required=True, help="Path to W2L architecture file"
    )
    parser.add_argument(
        "-nin",
        "--num_input_feat",
        type=int,
        default=80,
        help="number of input features",
    )
    parser.add_argument(
        "-nout",
        "--num_output_feat",
        type=int,
        default=30,
        help="number of target classes",
    )
    args = parser.parse_args()

    if not os.path.exists(args.arch):
        print("'" + args.arch + "' - file doesn't exist")
        exit()

    overall_kw = 1
    overall_dw = 1
    params = 0

    with open(args.arch, "r") as arch:
        for line in arch:
            line = line.replace("NLABEL", str(args.num_output_feat))
            line = line.replace("NFEAT", str(args.num_input_feat))
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            linesplit = line.split()
            if linesplit[0] == "WN":
                linesplit = linesplit[2:]

            layer = linesplit[0]
            kw = 1
            dw = 1
            dil = 1
            if layer == "L":
                inC = int(linesplit[1])
                outC = int(linesplit[2])
                params += inC * outC * kw
            elif layer == "C" or layer == "C1":
                inC = int(linesplit[1])
                outC = int(linesplit[2])
                kw = int(linesplit[3])
                dw = int(linesplit[4])
                if len(linesplit) > 6:
                    dil = int(linesplit[6])
                params += inC * outC * kw
            elif layer == "C2":
                inC = int(linesplit[1])
                outC = int(linesplit[2])
                kw = int(linesplit[3])
                kwy = int(linesplit[4])
                outC = int(linesplit[1])
                dw = int(linesplit[5])
                if len(linesplit) > 9:
                    dil = int(linesplit[9])
                params += inC * outC * kw * kwy
            elif layer == "CL":
                kw = int(linesplit[2])
                # strided/dilated ConvLinear layers only for now
            elif layer == "M" or layer == "A":
                kw = int(linesplit[1])
                dw = int(linesplit[3])
                # dilated pooling not supported
            elif layer == "TDS":
                c = int(linesplit[1])
                kw = int(linesplit[2])
                h = int(linesplit[3])
                params += c * c * (kw + 2 * h * h)

            overall_kw += (kw - 1) * overall_dw * dil
            overall_dw *= dw

print("---- Arch Stats ----")
print("Receptive Field (kw)   :", overall_kw)
print("Overall Stride (dw)    :", overall_dw)
print("Overall Params (approx):", params)
