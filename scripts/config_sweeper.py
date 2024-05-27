import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Union
from time import sleep
import subprocess
from itertools import product
import os
from datetime import datetime
import random
from pathlib import Path
import sys
import yaml

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.warning("use `1>` to redirect job logging to file.")
LOGGER.warning("The logging of this scripts is in stderr, not stdout. Use `2>` to redirect the stderr to a file.")
print = LOGGER.info


class DictAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.
        All elements inside '()' or '[]' are treated as iterable values.
        Args:
            val (str): Value string.
        Returns:
            list | tuple: The expanded list or tuple from the string.
        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.
            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count("(") == string.count(")")) and (
                string.count("[") == string.count("]")
            ), f"Imbalanced brackets exist in {string}"
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if (char == ",") and (pre.count("(") == pre.count(")")) and (pre.count("[") == pre.count("]")):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip("'\"").replace(" ", "")
        is_tuple = False
        if val.startswith("(") and val.endswith(")"):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        elif "," not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1 :]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            try:
                key, val = kv.split("=", maxsplit=1)
                options[key] = self._parse_iterable(val)
            except ValueError as e:
                print(f"Invalid option {kv}")
                raise e
        setattr(namespace, self.dest, options)


@dataclass
class GPUStatus:
    gpu_id: Union[int, List[int]]
    occupied: bool


def wait_until(process_queue: List, gpu_status: List[GPUStatus], sleep_time=10, verbose=True):
    # [TODO] event driven
    while True:
        return_flag = False
        for job_id, current_gpu_id, gpu_status_id, process in process_queue:
            if process.poll() is not None:
                returncode = process.poll()
                if returncode == 0:
                    msg = f"job {job_id}, finished!"
                else:
                    msg = f"job {job_id}, failed! return code: {returncode}"
                if verbose:
                    print(msg)
                return_flag = True
                gpu_status[gpu_status_id].occupied = False
                process_queue.remove((job_id, current_gpu_id, gpu_status_id, process))
        if return_flag is True:
            return (job_id, returncode)
        sleep(sleep_time)


def get_valid_gpu_id(gpu_status):
    for i, _gpu_status in enumerate(gpu_status):
        if _gpu_status.occupied is False:
            current_gpu_id = _gpu_status.gpu_id
            gpu_status_id = i
            _gpu_status.occupied = True
            return current_gpu_id, gpu_status_id
    return None, None

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

if __name__ == "__main__":
    parser = argparse.ArgumentParser("A simple job config sweeper")
    parser.add_argument(
        "--gpus", "-g", type=str, default="8", 
        help="int: the number of gpus, e.g. -g=8 means use gpu 0-7; tuple: specify gpu info per job, e.g., -g='(0, 0, (2,3), (1,5,6))'."
    )
    parser.add_argument("--script", "-s", required=True, help="the script to be run.")
    parser.add_argument("--config-name", "--config_name",type=str, default=None, help="the config name")
    parser.add_argument("--config-dir", "--config_dir", type=str, default=None, help="the config dir")
    parser.add_argument("--query_interval", "-i", type=int, default=10, help='The interval to query the status of the job. To be changed in the future.')
    parser.add_argument('--dry_run', '-d', action='store_true', help='Dry run, do not run the command.')
    parser.add_argument("--no_skip_existing", "-n", action="store_true", help="Skip existing files.")
    parser.add_argument("--random_order", "-r", action="store_true", help="Random order.")
    parser.add_argument('params', nargs='*', action=DictAction, help='The parameters to sweep, e.g., --params="lr=0.1,0.01,0.001" "batch_size=32,64,128"')
    parser.add_argument("--meta_config_file", "-f", type=str, default=None, help="The meta config file.")
    parser.add_argument("--torchrun", action='store_true')
    parser.add_argument("--offline", action='store_true')

    args = parser.parse_args()
    if args.dry_run:
        print('Dry run, do not run the command.')
        args.query_interval = 0.0
    print(f"args: {args}")

    # setup gpu
    print("Set up GPUs")
    try:
        gpus = int(args.gpus)
        gpus = list(range(gpus))
    except ValueError:
        raw_gpus = DictAction._parse_iterable(args.gpus)
        gpus = []
        
        for gpu in raw_gpus:
            if isinstance(gpu, (list, tuple)):
                gpus.append(",".join(list(map(str,gpu))))
            elif isinstance(gpu, int):
                gpus.append(str(gpu))
            else:
                raise ValueError(f"Invalid gpu id: {gpu}, type: {type(gpu)}")
        
    print(f"GPUs: {gpus}, num: {len(gpus)}")
    
    # gpus status init
    max_jobs_in_parallel = len(gpus)
    process_queue = []
    gpu_status = [GPUStatus(gpu_id=gpus[i % max_jobs_in_parallel], occupied=False) for i in range(max_jobs_in_parallel)]

    if args.meta_config_file is None:
        # setup jobs, convert one param into list
        for k, v in args.params.items():
            if not isinstance(v, (tuple, list)):
                args.params[k] = [v if v is not None else 'null']
        
        # setup jobs, get all combinations
        params_combinations = []
        values_combinations = list(product(*args.params.values()))
        keys = args.params.keys()
        for values in values_combinations:
            params = dict(zip(keys, values))
            params_cmd = " ".join([f"{k}={v if v is not None else 'null'}" for k, v in params.items()])
            params_combinations.append(params_cmd)
    else:
        params_combinations = []
        with open(args.meta_config_file, "r") as f:
            meta_config = yaml.safe_load(f)

        for meta_key, params in meta_config.items():
            print(f"meta_key: {meta_key}, params: {params}")
            for k, v in params.items():
                if not isinstance(v, (tuple, list)):
                    params[k] = [v if v is not None else 'null']

            # append command line params
            for k, v in args.params.items():
                if not isinstance(v, (tuple, list)):
                    params[k] = [v if v is not None else 'null']

            keys = params.keys()
            values_combinations = list(product(*params.values()))
            for values in values_combinations:
                params = dict(zip(keys, values))
                params_cmd = " ".join([f"{k}={v if v is not None else 'null'}" for k, v in params.items()])
                params_combinations.append(params_cmd)

    if args.random_order:
        print(f"Random order: {args.random_order}")
        random.shuffle(params_combinations)
    
    print(f"number of jobs {len(params_combinations)}")

    # run jobs
    num_jobs = 0
    job_cmds = []
    out_dirs = []
    failed_jobs = 0

    for i, params in enumerate(params_combinations):
        out_dir_cmd = " ".join(["python scripts/hydra_config_get_out_dir.py", (f"--config-name {args.config_name}" if args.config_name else ""), (f"--config-dir {args.config_dir}" if args.config_dir else ""), f"{params}"])
        out_dir = subprocess.check_output(out_dir_cmd, shell=True).decode('utf-8').strip()
        out_dir = os.path.dirname(out_dir)  # remove the last slash, e.g. version_0/
        # [XXX] remove empty dir made by hydra for out_dir finding.
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) == 0:
            os.rmdir(out_dir)

        fail_processes = list(Path(out_dir).glob("config_sweeper.fail"))
        if len(fail_processes) > 0:
            # check if the fail file is created by this script
            print(f"Found unfinished processes: {fail_processes} , re-run.")
            if not args.dry_run:
                for fail_process in fail_processes:
                    fail_process.unlink()
        elif not args.no_skip_existing and os.path.exists(out_dir):
            # no failed jobs, check if the job is already done
            cmd = f'python {args.script} {(f"--config-name {args.config_name}" if args.config_name else "")} {(f"--config-dir {args.config_dir}" if args.config_dir else "")} {params}'
            print(f"Skip existing job: {cmd}")
            continue

        if len(process_queue) == max_jobs_in_parallel:
            # print(f"Reach max jobs in parallel {max_jobs_in_parallel}, wait until one job finished")
            ret_job_id, ret_code = wait_until(process_queue, gpu_status, args.query_interval, verbose=not args.dry_run)
            if ret_code != 0:
                with open(f"{out_dirs[ret_job_id]}/config_sweeper.fail", "a") as f:
                    f.write(f"[{datetime.now().strftime('%y/%m/%d %H:%M:%S')}]\
                            Job: {ret_job_id} failed. Command: {job_cmds[ret_job_id]}. Return code{ret_code}\n")
                failed_jobs += 1

        current_gpu_id, gpu_status_id = get_valid_gpu_id(gpu_status)
        if isinstance(current_gpu_id, str):
            num_gpus_current_job = str(len(current_gpu_id.split(",")))
        else:
            num_gpus_current_job = "1"
        cmd = f'WANDB_MODE={"online" if not args.offline else "offline"} CUDA_VISIBLE_DEVICES={current_gpu_id} {"python" if not args.torchrun else "torchrun --nproc_per_node " + num_gpus_current_job + " --master_port " + str(_find_free_port())} {args.script} {(f"--config-name {args.config_name}" if args.config_name else "")} {(f"--config-dir {args.config_dir}" if args.config_dir else "")} {params}'
        print(f"[Jobs]:\t{i}\t[CMD]:\t{cmd}\t[OUT_DIR]:\t{out_dir}")

        if not args.dry_run:
            sleep(random.random() / 2 + args.query_interval)
            process = subprocess.Popen(cmd, shell=True)
        else:
            process = subprocess.Popen("echo 'sleep in dry_run'; sleep 0", shell=True)
        process_queue.append((i, current_gpu_id, gpu_status_id, process))

        job_cmds.append(cmd)
        out_dirs.append(out_dir)
        num_jobs += 1

    while len(process_queue) != 0:
        ret_job_id, ret_code = wait_until(process_queue, gpu_status)
        if ret_code != 0:
            with open(f"{out_dirs[ret_job_id]}/config_sweeper.fail", "a") as f:
                f.write(f"[{datetime.now().strftime('%y/%m/%d %H:%M:%S')}]\
                        Job: {ret_job_id} failed. Command: {job_cmds[ret_job_id]}. Return code{ret_code}\n")
            failed_jobs += 1
        
    print(f"Total jobs: {num_jobs}, successed jobs {num_jobs - failed_jobs}, failed jobs: {failed_jobs}")
