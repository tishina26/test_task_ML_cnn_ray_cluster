# https://docs.ray.io/en/latest/cluster/kubernetes/examples/ml-example.html#submit-the-workload

import subprocess

if __name__ == "__main__":
    print("Make sure all files are in the local directory:")
    print("\"ray_light.py\", \"ray_light_cnn.py\", \"using_sbatch_and_ray.py\", \"template.sh\"")

    FILE_1 = "ray_light.py"
    FILE_2 = "ray_light_cnn.py"
    RUN_TEMPLATE = "python using_sbatch_and_ray.py --num-cpus {cpus} --priority {priority} --command \"python {file}\""

    p1 = subprocess.Popen(RUN_TEMPLATE.format(
        cpus=4,
        priority=-1,
        file=FILE_1
    ))

    p2 = subprocess.Popen(RUN_TEMPLATE.format(
        cpus=1,
        priority=-10,
        file=FILE_2
    ))

    for i in p1, p2:
        i.wait()

    print("Succeeded to make models")





