#! /home/vafaeisa/miniconda3/bin/python

import argparse
import os
from pathlib import Path
import subprocess

import pyslurm


# Make sure to clone/fetch your repository
SYNC_SCRIPT_PATH = '/home/vafaeisa/sync_mlflow.sh'

USER = os.getenv('USER')
MAX_JOBS = 10
mj = os.getenv('MAX_JOBS')
if mj is not None:
    MAX_JOBS = int(mj)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_root',
                        type=str,
                        help='**absolute** directory of the project repository.',
                        required=True)

    parser.add_argument('--run_name',
                        type=str,
                        help='repo_root_dir/runs/run_name',
                        required=True)

    parser.add_argument('--email',
                        type=str,
                        help='your job-events will be sent to this address',
                        required=True)

    parser.add_argument('--data_dir',
                        type=str,
                        help='**absolute** directory of the dataset',
                        required=False)

    parser.add_argument('--gpu_type',
                        type=str,
                        help='either p100, v100 or t4',
                        required=False,
                        default='p100')

    parser.add_argument('--hours',
                        type=int,
                        help='minimum required hours',
                        required=False,
                        default=2)

    parser.add_argument('--mem',
                        type=int,
                        help='minimum required memory in megabytes',
                        required=False,
                        default=12288)

    parser.add_argument('--n_cpu',
                        type=int,
                        help='cpus to use',
                        required=False,
                        default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    jobs = pyslurm.job().find_user(USER)
    if len(jobs) >= MAX_JOBS:
        print(f'job queue is full, try again later.')
    else:
        project_root = Path(args.project_root).absolute()
        assert project_root.is_dir(), f"{project_root} is not a directory."

        run_name = args.run_name
        run_dir = project_root.joinpath('runs').joinpath(run_name)
        assert run_dir.is_dir(), f'run directory {run_dir} does not exist'

        job_name = f'{project_root.name}_{args.run_name}'

        email = args.email
        memory = args.mem
        hours = args.hours
        gpu_type = args.gpu_type
        n_cpu = args.n_cpu

        data_dir = args.data_dir
        if data_dir is not None:
            assert Path(data_dir).is_dir(), f'data directory {data_dir} does not exist'

        job_script_path = run_dir.joinpath(job_name + '.job')
        err_log_path = run_dir.joinpath(f'log-{job_name}.err')
        out_log_path = run_dir.joinpath(f'log-{job_name}.out')

        with open(job_script_path, 'w') as f:
            f.write('#!/bin/env bash\n\n')

            f.write('#SBATCH --get-user-env\n')
            f.write(f'#SBATCH --gres gpu:{gpu_type}:1\n')
            f.write(f'#SBATCH --job-name {job_name}\n')
            f.write(f'#SBATCH --time {hours}:00:00\n')
            f.write(f'#SBATCH --output {out_log_path}\n')
            f.write(f'#SBATCH --error {err_log_path}\n')
            f.write(f'#SBATCH --mem {memory}\n')
            f.write(f'#SBATCH --mail-user {email}\n')
            f.write(f'#SBATCH --cpus-per-task={n_cpu}\n')
            f.write(f'#SBATCH --mail-type ALL\n\n')

            # f.write(f'module load cuda/11.0\n')
            # f.write(f'module load cudnn\n\n')

            f.write('export CUDA_HOME=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.2.2\n')
            f.write('export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}\n')
            # f.write('export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH\n\n')
            f.write('export CUDA_HDIR=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.2.2\n')
            f.write('export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/libdevice${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n\n')

            f.write(f'cd {project_root}\n')

            if data_dir is None:  # Use config.data_dir
                f.write(f'train --run_name {run_name}\n --project_root {project_root}\n')
            else:
                f.write(f'train --run_name {run_name} --data_dir {str(data_dir)} --project_root {project_root}\n')

        # os.system(f'sbatch {job_script_path}')
        submit_cmd = ['sbatch',  str(job_script_path)]
        res = subprocess.check_output(submit_cmd).strip()
        if not res.startswith(b"Submitted batch"):
            print(f'could not submit the batch: {res}')
        else:
            job_id = int(res.split()[-1])
            print(f'submitted job {job_id}')
            # os.system(f'strigger --set --jobid={job_id} --fini --program={SYNC_SCRIPT_PATH}')


if __name__ == '__main__':
    main()
