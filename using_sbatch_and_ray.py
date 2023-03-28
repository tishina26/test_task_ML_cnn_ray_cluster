# https://github.com/pengzhenghao/use-ray-with-slurm/blob/main/launch.py

# Usage: python using_sbatch_and_ray.py --num-cpus 2 --command "python ray_light.py" --priority -1

import argparse
import subprocess
import sys
import time

from pathlib import Path

template_file = Path(__file__) / 'template.sh'

NUM_CPUS = "{{NUM_CPUS}}"
JOB_PRIORITY = "{{JOB_PRIORITY}}"
COMMAND_PLACEHOLDER = "{{COMMAND_PLACEHOLDER}}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-cpus", "-n", type=int, default=1,
        help="Number of cpus to use."
    )
    parser.add_argument(
        "--command", type=str, required=True,
        help="The command you wish to execute. For example: --command 'python "
             "test.py' Note that the command must be a string."
    )
    parser.add_argument(
        "--priority", type=int, default=0,
        help="Priority of submitted job. Default = 0"
    )
    args = parser.parse_args()

    if args.node:
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}".format(
        time.strftime("%m%d-%H%M", time.localtime())
    )

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(NUM_CPUS, str(args.num_cpus))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(JOB_PRIORITY, str(args.priority))
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!"
    )

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Start to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name))
    )
    sys.exit(0)
