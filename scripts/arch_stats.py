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
    args = parser.parse_args()

    if not os.path.exists(args.arch):
        print("'" + args.arch + "' - file doesn't exist")
        exit()

    overall_kw = 1
    overall_dw = 1

    with open(args.arch, "r") as arch:
        for line in arch:
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
            if layer == "C" or layer == "C1":
                kw = int(linesplit[3])
                dw = int(linesplit[4])
                if len(linesplit) > 6:
                    dil = int(linesplit[6])
            elif layer == "C2":
                kw = int(linesplit[3])
                dw = int(linesplit[5])
                if len(linesplit) > 9:
                    dil = int(linesplit[9])
            elif layer == "CL":
                kw = int(linesplit[2])
                # strided/dilated ConvLinear layers only for now
            elif layer == "M" or layer == "A":
                kw = int(linesplit[1])
                dw = int(linesplit[3])
                # dilated pooling not supported

            overall_kw += (kw - 1) * overall_dw * dil
            overall_dw *= dw

print("---- Arch Stats ----")
print("Receptive Field (kw) :", overall_kw)
print("Overall Stride (dw)  :", overall_dw)
