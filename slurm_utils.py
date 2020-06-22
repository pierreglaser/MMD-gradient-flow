import pprint
import logging
import os
import re
import subprocess

from dask_jobqueue import SLURMCluster
from distributed import Client

CPUS_PER_NODE = 24

def get_sbatch_args(n_workers, proc_per_worker):
    """generates the slurm args to use an optimal number of nodes given the
    number of requested workers
    """
    num_nodes = (n_workers * proc_per_worker) // CPUS_PER_NODE + 1

    # cmd = f"sinfo -N | grep node | grep idle | grep cpu | cut -d' ' -f 1"
    cmd = f"sinfo -N | grep cpu"
    all_nodes = subprocess.check_output(cmd, shell=True)
    all_nodes = all_nodes.decode().strip().split(' \n')

    # __import__('pdb').set_trace()
    all_nodes = [re.split(r'\s+', n.strip(' ')) for n in all_nodes]
    assert all(len(n) == 4 for n in all_nodes)
    idle_cpu_nodes_names = [name for name, _, hardware, state in all_nodes if state == 'idle' and 'cpu' in hardware]

    assert len(idle_cpu_nodes_names) >= num_nodes
    nodes_names_to_use = idle_cpu_nodes_names[-num_nodes:]
    # nodes_name_to_exclude = [n[5:] for n, _, _, _ in all_nodes if n not in nodes_names_to_use]
    nodes_name_to_exclude = [n for n, _, _, _ in all_nodes if n not in nodes_names_to_use]

    excluded_nodes_str = ','.join(nodes_name_to_exclude)
    print(excluded_nodes_str)
    print(f'will use nodes: {nodes_names_to_use}')
    # each worker is a batch script, tell slurm how many cpu the dask worker uses
    # for efficient load balancing
    return f'-x {excluded_nodes_str} --cpus-per-task={proc_per_worker} --ntasks=1'
