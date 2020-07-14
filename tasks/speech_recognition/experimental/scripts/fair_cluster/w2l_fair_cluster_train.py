#!/usr/bin/env python3

import os
import pathlib
import shutil
import stat
import subprocess


W2L_ROOT = "/private/home/${USER}/wav2letter"
LOG_ROOT = "/checkpoint/qiantong/ls_200M"


def _sbatch(command, run_sh_path, dry_run, **kwargs):
    def dict_to_args(kwargs):
        args = []
        for k, v in kwargs.items():
            args.append("--" + k)
            args.append(str(v))
        return args

    args = ["sbatch"]
    args.extend(dict_to_args(kwargs))
    args.append("--wrap")
    args.append(f"set -x; srun --label {run_sh_path}")
    print("Command:\n\t", command)
    print("Sbatch:\n\t", args)
    if not dry_run:
        subprocess.check_call(args)
    else:
        print("DRY RUN!", kwargs["job-name"])


def _get_kwargs(
    partition="learnfair",
    cpus_per_task=10,
    gpus=16,
    gpu32=False,
    mem_per_gpu=8,
    hours=72,
):
    assert gpus > 0
    assert gpus <= 8 or gpus % 8 == 0, gpus

    kwargs = {}
    ntasks = min(8, gpus)
    kwargs["cpus-per-task"] = cpus_per_task
    kwargs["partition"] = partition
    kwargs["gres"] = f"gpu:volta:{ntasks}"
    if gpu32:
        kwargs["gres"] = f"gpu:volta32gb:{ntasks}"
    kwargs["ntasks-per-node"] = ntasks
    kwargs["nodes"] = max(1, gpus // 8)
    kwargs["time"] = f"{hours}:0:0"
    kwargs["mem"] = f"{mem_per_gpu * max(1, ntasks)}GB"
    kwargs["signal"] = "USR1@60"
    return kwargs


def main(config_path, dry_run, mode, **kwargs):
    gpus = kwargs["gpus"]
    assert config_path.exists(), config_path
    kwargs = _get_kwargs(**kwargs)

    # Train mode
    if mode == "train":
        exp_name = config_path.parent.name
        exp_id, _ = config_path.name.rsplit(".", 1)

        command = " ".join(
            f"""
                {W2L_ROOT}/build/Train train
                --flagsfile={config_path}
                --enable_distributed
                --logtostderr=1
        """.split()
        )

        assert os.path.exists(LOG_ROOT)
        log_dir = os.path.join(LOG_ROOT, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sub_log_dir = os.path.join(log_dir, exp_id)
        if os.path.exists(sub_log_dir):
            print("Exists:", sub_log_dir)
            print("Skipping")
            return

        rndv_dir = os.path.join(log_dir, exp_id + "_rndv")
        if not os.path.exists(rndv_dir):
            os.makedirs(rndv_dir)
        command += f" --rndv_filepath={rndv_dir}"
        command += f" --runname={exp_id} --rundir={log_dir}"

    # Continue mode
    elif mode == "continue":
        log_dir = config_path.parent
        exp_name = config_path.parent.name
        exp_id = config_path.name

        command = f"{W2L_ROOT}/build/Train continue {config_path}"

        rndv_dir = os.path.join(log_dir, exp_id + "_rndv")
        shutil.rmtree(rndv_dir)
        os.makedirs(rndv_dir)

    # WTF
    else:
        print("Not supported mode: ", mode)
        print("Skipping")
        return

    command += f" --world_size={gpus}"
    command += " --world_rank=${SLURM_PROCID}"

    kwargs.update(
        {
            "comment": "ICASSP 2020 deadline",
            "job-name": f"{exp_name}:{exp_id}",
            "open-mode": "append",
            "output": f"{log_dir}/{exp_id}.out",
            "error": f"{log_dir}/{exp_id}.err",
        }
    )

    run_sh_path = os.path.join(log_dir, f"{exp_id}_{mode}.sh")
    with open(run_sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(command)
    st = os.stat(run_sh_path)
    os.chmod(run_sh_path, st.st_mode | stat.S_IEXEC)

    _sbatch(command, run_sh_path, dry_run, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=pathlib.Path)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--partition", type=str, default="learnfair")
    parser.add_argument("--cpus_per_task", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--gpu32", action="store_true")
    parser.add_argument("--mem_per_gpu", type=int, default=20)
    parser.add_argument("--hours", type=int, default=72)
    main(**vars(parser.parse_args()))
