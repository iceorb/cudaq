#!/usr/bin/env python3

"""
GPU-Aware Distributed Job Dispatcher with Persistent Tracking
=============================================================
A Python program that monitors available GPU memory (via pynvml) and dispatches
jobs to the GPU with the most free memory, subject to a minimum memory threshold.
Jobs can be run in parallel on multiple GPUs, and all job assignments and states
are tracked in a local JSONL file for persistence.

Features:
---------
1. Monitors available GPU memory and assigns jobs to the GPU with the most free memory.
2. Optionally specify which GPU IDs are valid for job assignment.
3. Dispatches jobs in parallel, each with CUDA_VISIBLE_DEVICES set appropriately.
4. If no GPU meets the memory threshold, the dispatcher waits and retries.
5. Tracks job states in a JSONL file: (pid, start time, status, assigned GPU, command, etc.).
6. Recovers state after restart by reloading the JSONL, detecting running processes, and resuming.
7. Logs each job's stdout/stderr to a dedicated log file and tracks overall assignments in a master log.
8. Provides a "status" CLI command to summarize all jobs.

Usage:
------
1. Install dependencies:
    pip install pynvml pyyaml

2. Prepare a commands file (e.g., commands.txt), one shell command per line.

3. Run dispatcher:
    python cudaq.py run --commands-file commands.txt

4. Check status:
    python cudaq.py status

5. Optionally create a config YAML (see example below) and reference it:
    python cudaq.py run --config config.yaml

Example Config File (config.yaml):
----------------------------------
gpu_ids: [0, 1]             # Use only GPU 0 and 1
min_free_mem_mb: 6000       # Minimum free memory requirement
poll_interval: 60           # Check GPU memory every 60s
commands_file: commands.txt # File containing shell commands
log_dir: ./logs             # Directory for job logs
jobs_file: ./jobs.jsonl     # JSONL file to track jobs
"""

import os
import sys
import time
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import pynvml
except ImportError:
    print("Please install pynvml (pip install pynvml) to use this dispatcher.")
    sys.exit(1)

DEFAULT_CONFIG = {
    "gpu_ids": None,             # If None, use all available GPUs
    "min_free_mem_mb": 6000,     # Minimal free memory required to start a job on a GPU
    "poll_interval": 60,         # How often (in seconds) to poll GPU memory if no GPU is available
    "commands_file": "commands.txt",
    "log_dir": "./logs",
    "jobs_file": "./jobs.jsonl",
}


def load_config(config_file=None, cli_args=None):
    """
    Load configuration from a YAML file (if provided), then override
    with command-line arguments, and fill in defaults where not specified.
    """
    config = DEFAULT_CONFIG.copy()

    if config_file and os.path.exists(config_file):
        with open(config_file, "r") as f:
            file_config = yaml.safe_load(f)
        if file_config:
            config.update(file_config)

    # Override config with CLI arguments (if given)
    if cli_args.gpu_ids is not None:
        # e.g. --gpu-ids "0,1,3"
        config["gpu_ids"] = [int(x) for x in cli_args.gpu_ids.split(",")]
    if cli_args.min_free_mem_mb is not None:
        config["min_free_mem_mb"] = int(cli_args.min_free_mem_mb)
    if cli_args.poll_interval is not None:
        config["poll_interval"] = int(cli_args.poll_interval)
    if cli_args.commands_file is not None:
        config["commands_file"] = cli_args.commands_file
    if cli_args.log_dir is not None:
        config["log_dir"] = cli_args.log_dir
    if cli_args.jobs_file is not None:
        config["jobs_file"] = cli_args.jobs_file

    return config


def init_pynvml():
    """Initialize NVML to query GPU memory info."""
    pynvml.nvmlInit()


def get_gpu_free_mem_mb(gpu_id):
    """
    Return free memory (in MB) for the given GPU ID using NVML.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mem_mb = mem_info.free / 1024 / 1024
    return free_mem_mb


def get_all_gpus_free_mem_mb(gpu_ids=None):
    """
    Return a dict {gpu_id: free_mem_mb, ...}.
    If gpu_ids is None, queries all GPUs in the system.
    """
    device_count = pynvml.nvmlDeviceGetCount()
    if gpu_ids is None:
        gpu_ids = range(device_count)
    mem_dict = {}
    for gid in gpu_ids:
        try:
            mem_dict[gid] = get_gpu_free_mem_mb(gid)
        except pynvml.NVMLError as e:
            # If we cannot query this GPU, skip it
            print(f"[Warning] Failed to query GPU {gid}: {e}")
    return mem_dict


def load_jobs(jobs_file):
    """
    Load existing jobs from a JSONL file. Each line is a JSON object representing a job.
    Returns a list of dicts.
    """
    if not os.path.exists(jobs_file):
        return []

    jobs = []
    with open(jobs_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    job_data = json.loads(line)
                    jobs.append(job_data)
                except json.JSONDecodeError:
                    print(f"[Warning] Invalid JSON line in {jobs_file}: {line}")
    return jobs


def save_job(jobs_file, job_data):
    """
    Append a job record to the JSONL file.
    """
    with open(jobs_file, "a") as f:
        f.write(json.dumps(job_data) + "\n")


def rewrite_jobs_file(jobs_file, all_jobs):
    """
    Overwrite the JSONL file with the updated list of job records.
    """
    with open(jobs_file, "w") as f:
        for job in all_jobs:
            f.write(json.dumps(job) + "\n")


def is_process_running(pid):
    """
    Check if a process with the given PID is still running.
    """
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def dispatch_job(job, gpu_id, config):
    """
    Dispatch a single job command with CUDA_VISIBLE_DEVICES set to 'gpu_id'.
    Logs to a dedicated file under config["log_dir"].
    Returns the updated job dictionary with pid, start_time, etc.
    """
    Path(config["log_dir"]).mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config["log_dir"], f"job_{timestamp_str}.log")

    # Start the subprocess with environment variable for GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # We redirect stdout/stderr to the log file
    with open(log_file, "w") as f:
        # Start the process
        process = subprocess.Popen(
            job["command"],
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )

    job["gpu_id"] = gpu_id
    job["log_file"] = log_file
    job["start_time"] = datetime.now().isoformat()
    job["pid"] = process.pid
    job["status"] = "running"
    return job


def run_cudaq(config):
    """
    Main logic for the dispatcher. Continuously checks if there are pending jobs,
    assigns them to GPUs with enough free memory, and updates job statuses.
    """
    jobs_file = config["jobs_file"]
    all_jobs = load_jobs(jobs_file)

    # 1) Detect running vs. completed jobs from previous session
    for job in all_jobs:
        if job["status"] == "running":
            pid = job.get("pid")
            if not is_process_running(pid):
                # Mark as completed or failed (we don't know exit code, but let's guess)
                job["status"] = "failed"
    rewrite_jobs_file(jobs_file, all_jobs)

    # 2) If the user provided a commands_file, load new commands as new "pending" jobs
    if os.path.exists(config["commands_file"]):
        with open(config["commands_file"], "r") as f:
            new_commands = [line.strip() for line in f if line.strip()]

        for cmd in new_commands:
            # Check if command is not already in jobs
            found = any(j["command"] == cmd for j in all_jobs)
            if not found:
                job_data = {
                    "command": cmd,
                    "status": "pending",
                    "pid": None,
                    "gpu_id": None,
                    "start_time": None,
                    "log_file": None,
                }
                all_jobs.append(job_data)
                save_job(jobs_file, job_data)

    print("[cudaq] Starting job dispatch loop.")
    init_pynvml()  # Initialize NVML for GPU queries

    while True:
        # Reload jobs each cycle (in case we updated externally or crashed)
        all_jobs = load_jobs(jobs_file)

        # Update status of running jobs
        changed = False
        for job in all_jobs:
            if job["status"] == "running":
                if not is_process_running(job["pid"]):
                    # Optionally retrieve return code
                    retcode = None
                    try:
                        retcode = os.waitpid(job["pid"], os.WNOHANG)[1]
                    except ChildProcessError:
                        pass
                    if retcode == 0:
                        job["status"] = "completed"
                    else:
                        job["status"] = "failed"
                    changed = True

        if changed:
            rewrite_jobs_file(jobs_file, all_jobs)

        # Check for pending jobs
        pending_jobs = [j for j in all_jobs if j["status"] == "pending"]
        if not pending_jobs:
            print(f"[cudaq] No pending jobs. Sleeping {config['poll_interval']}s ...")
            time.sleep(config["poll_interval"])
            continue

        # If there are pending jobs, see if we can dispatch them
        free_mem_dict = get_all_gpus_free_mem_mb(config["gpu_ids"])
        if not free_mem_dict:
            # If no valid GPUs, just wait
            print("[cudaq] No valid GPUs found. Check your GPU IDs. Sleeping...")
            time.sleep(config["poll_interval"])
            continue

        assigned_any_job = False
        for job in pending_jobs:
            # Pick the GPU with the maximum free memory that is >= min_free_mem_mb
            suitable_gpus = [(gid, mem) for gid, mem in free_mem_dict.items()
                             if mem >= config["min_free_mem_mb"]]
            if not suitable_gpus:
                # No GPU meets the requirement for this job; break and wait
                break
            suitable_gpus.sort(key=lambda x: x[1], reverse=True)
            best_gpu, best_gpu_mem = suitable_gpus[0]

            # Dispatch job to best_gpu
            print(f"[cudaq] Assigning job '{job['command']}' to GPU {best_gpu} "
                  f"(free memory ~ {int(best_gpu_mem)} MB).")
            updated_job = dispatch_job(job, best_gpu, config)
            idx = all_jobs.index(job)
            all_jobs[idx] = updated_job
            rewrite_jobs_file(jobs_file, all_jobs)

            # Subtract some memory estimate from that GPU
            free_mem_dict[best_gpu] -= config["min_free_mem_mb"]
            assigned_any_job = True

        if not assigned_any_job:
            print("[cudaq] Unable to dispatch any pending jobs; waiting...")
        time.sleep(config["poll_interval"])


def print_status(config):
    """
    Print the current state of all jobs in a readable format.
    """
    jobs = load_jobs(config["jobs_file"])
    for job in jobs:
        cmd = job.get("command", "")
        gpu_id = job.get("gpu_id", None)
        status = job.get("status", "unknown")
        pid = job.get("pid", None)

        icon = "[ ]"
        if status == "completed":
            icon = "[✓]"
        elif status == "running":
            icon = "[→]"
        elif status == "failed":
            icon = "[!]"
        elif status == "pending":
            icon = "[ ]"

        gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "GPU N/A"
        if status == "running":
            print(f"{icon} {cmd} → {gpu_str} → Running (PID {pid})")
        else:
            print(f"{icon} {cmd} → {gpu_str} → {status.capitalize()}")


def main():
    parser = argparse.ArgumentParser(description="GPU-Aware Job Dispatcher with Persistent Tracking")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run dispatcher
    parser_run = subparsers.add_parser("run", help="Run the dispatcher loop.")
    parser_run.add_argument("--config", help="Path to YAML config file", default=None)
    parser_run.add_argument("--gpu-ids", help="Comma-separated GPU IDs to use", default=None)
    parser_run.add_argument("--min-free-mem-mb", help="Minimum free memory in MB to dispatch a job", default=None)
    parser_run.add_argument("--poll-interval", help="Time in seconds between checks", default=None)
    parser_run.add_argument("--commands-file", help="Path to file with commands to run", default=None)
    parser_run.add_argument("--log-dir", help="Directory to store job logs", default=None)
    parser_run.add_argument("--jobs-file", help="Path to JSONL file tracking jobs", default=None)

    # status
    parser_status = subparsers.add_parser("status", help="Print status of all jobs.")
    parser_status.add_argument("--config", help="Path to YAML config file", default=None)
    parser_status.add_argument("--gpu-ids", help="Comma-separated GPU IDs to use", default=None)
    parser_status.add_argument("--min-free-mem-mb", help="Minimum free memory in MB to dispatch a job", default=None)
    parser_status.add_argument("--poll-interval", help="Time in seconds between checks", default=None)
    parser_status.add_argument("--commands-file", help="Path to file with commands to run", default=None)
    parser_status.add_argument("--log-dir", help="Directory to store job logs", default=None)
    parser_status.add_argument("--jobs-file", help="Path to JSONL file tracking jobs", default=None)

    args = parser.parse_args()
    config = load_config(args.config, args)

    if args.command == "run":
        run_cudaq(config)
    elif args.command == "status":
        print_status(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
