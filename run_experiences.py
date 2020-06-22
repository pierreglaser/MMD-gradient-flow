import itertools
import socket
import threading
import queue

from pathlib import Path

import pandas as pd

from distributed import Client, LocalCluster
from dask.distributed import as_completed
from dask_jobqueue import SLURMCluster

from train_student_teacher import parser, run_mmd_flow, _get_results_file
from slurm_utils import get_sbatch_args


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
    if socket.gethostname() == 'sgw1':

        # number of processing units per node. for ease of use, cores to the
        # number of CPU per node warning: this is the unitary increment by
        # which you can scale your number of workers inside your cluster.
        proc_per_worker = 24

        # total number of slurm node to request. Max number of dask workers
        # will be proc_per_worker * max_slurm_nodes
        max_slurm_nodes = 4

        cluster = SLURMCluster(
            workers=0,  # number of (initial slurm jobs)
            memory="16GB",
            # cores = number processing units per worker, can be
            # dask.Worker (processes) or threads of a worker's
            # ThreadPoolExecutor
            cores=proc_per_worker,
            # among those $cores workers, how many should be dask Workers,
            # (each worker will then have cores // processes threads inside
            # their ThreadPoolExecutor)
            # sets cpus-per-task=processes inside batch script
            processes=proc_per_worker,
            # job_extra=[get_sbatch_args(max_workers, proc_per_worker)],
        )
        # scale the number of unitary dask workers (and not batch jobs)
        cluster.scale(96)
    else:
        cluster = LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False,
            dashboard_address=':7777'
        )
    return cluster


def _make_result_df_entry(result, result_id, make_df=False):
    names = sorted(result["metadata"]) + ["commit_hash", "time", "id"]

    idx = [result["metadata"][k] for k in sorted(result["metadata"])] + [
        result["commit_hash"],
        result["time"],
        int(result_id),
    ]
    df = pd.DataFrame(result["records"])
    df.index.name = 'iter_no'
    if make_df:
        return pd.concat([df], keys=(tuple(idx),), names=names)
    else:
        return df, idx, names


def _write_result(result_queue):
    write_done = 0
    while write_done < len(fs):
        result = result_queue.get()
        result = _make_result_df_entry(result, write_done, make_df=True)

        file_ = _get_results_file(Path('./results'), extension='csv')

        if write_done == 0:
            result.to_csv(file_, mode='w', header=True)
        else:
            result.to_csv(file_, mode='a', header=False)

        write_done += 1
        print('wrote result no', write_done)


if __name__ == "__main__":
    n_epochs = 10000

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

    result_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=_write_result, args=(result_queue, )
    )

    for seed, entry, noise_level, dim, noise_in_prediction in all_configs:
        args = format_args(
            seed=seed, noise_level=noise_level, dim=dim, **entry
        )
        args += noise_in_prediction
        args = parser.parse_args(args.split(' '))
        f = client.submit(run_mmd_flow, args)
        fs.append(f)

    writer_thread.start()

    done = 0
    for future in as_completed(fs, raise_errors=False):
        try:
            result = future.result()
            done += 1
            print('number of runs lefts: {}'.format(len(all_configs) - done))
            result_queue.put(result)
        except Exception:
            print('something wrong happened for result', done)
            continue

    print('all results done, joining writer thread...')
    writer_thread.join()
