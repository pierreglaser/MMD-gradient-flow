import itertools
import socket

from pathlib import Path
from distributed import Client, LocalCluster
from dask.distributed import as_completed

from train_student_teacher import parser, run_mmd_flow, _append_to_results_file
from distributed_slurm import get_sbatch_args

from dask_jobqueue import SLURMCluster

N = 100


def format_args(N, non_linearity, lr, n_epochs, seed, noise_level, dim):
    args = (
        f"--total_epochs={n_epochs} "
        f"--N_train={N} --N_valid={N} --batch_size={N} "
        f"--num_particles=50 "
        f"--H=3 --seed={seed} --noise_level={noise_level} --d_int={dim} "
        f"--lr={lr} "
        f"--non_linearity={non_linearity} "
        f"--log_level=WARNING"
    )
    return args


def make_cluster():
    if socket.gethostname() != 'sgw1':
        proc_per_worker = 4
        max_workers = 20
        cluster = SLURMCluster(
            workers=0,  # number of (initial slurm jobs)
            memory="16GB",
            # cores = number of dask Worker processes lauched. I want only 1
            # dask.Worker per distributed worker.
            cores=1,
            # arguments to dask-worker CLI
            extra=[f'--nthreads {proc_per_worker} --nprocs=1'],
            # arguments to sbatch
            job_extra=[get_sbatch_args(max_workers, proc_per_worker)],
        )
        cluster.scale(20)
    else:
        cluster = LocalCluster(
            workers=2, memory="2GB", cores=1, threads_per_worker=1
        )
    return cluster


if __name__ == "__main__":
    n_epochs = 100

    entries = [
        {"n_epochs": n_epochs, "N": 100, "non_linearity": "quadexp", "lr": 0.1},  # noqa
        {"n_epochs": n_epochs, "N": 100, "non_linearity": "power", "lr": 0.0001},  # noqa
        {"n_epochs": n_epochs, "N": 100, "non_linearity": "cosine", "lr": 0.0001},  # noqa
        {"n_epochs": n_epochs, "N": 100, "non_linearity": "laplace", "lr": 0.1},  # noqa
        {"n_epochs": n_epochs, "N": 100, "non_linearity": "imq", "lr": 0.1},  # noqa
    ]

    seeds = range(5)
    noise_levels = [0, 1]
    dims = [2, 20, 50, 100]
    noise_in_predictions = [" --inject_noise_in_prediction", ""]

    i = 0

    cluster = make_cluster()
    client = Client(cluster)

    fs = []

    all_configs = list(itertools.product(
        seeds, entries, noise_levels, dims, noise_in_predictions
    ))
    print('total number of configs', len(all_configs))
    for seed, entry, noise_level, dim, noise_in_prediction in all_configs:
        args = format_args(
            seed=seed, noise_level=noise_level, dim=dim, **entry
        )
        args += noise_in_prediction
        args = parser.parse_args(args.split(' '))
        f = client.submit(run_mmd_flow, args)
        fs.append(f)

    done = 0
    for future in as_completed(fs, raise_errors=False):
        try:
            result = future.result()
            done += 1
            print('number of runs lefts: {}'.format(len(all_configs) - done))
            _append_to_results_file(result, Path('./results'))
        except Exception:
            continue
