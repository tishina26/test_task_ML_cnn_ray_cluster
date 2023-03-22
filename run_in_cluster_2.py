# https://docs.ray.io/en/latest/cluster/kubernetes/examples/ml-example.html#submit-the-workload

from ray.job_submission import JobSubmissionClient, JobStatus
import time

client = JobSubmissionClient("http://127.0.0.1:8265")

def run_in_cluster(CPUS, FILE_TO_RUN):

    place = (
        "git clone https://github.com/tishina26/test_task;"
        "python tishina26/test_task/" + FILE_TO_RUN
    )

    id = client.submit_job(
        entrypoint=place,
        entrypoint_num_cpus=CPUS
    )
    return id

def wait_until_status(id, status, timeout):
    start = time.time()
    while (client.get_job_status(id) not in status):
        if (time.time() - start > timeout):
            print("Failed to wait until status " + str(status))
            return
        time.sleep(10)

if __name__ == "__main__":
    FILE_1 = "ray_light.py"
    FILE_2 = "ray_light_cnn.py"
    TIMEOUT = 60

    id_1 = run_in_cluster(4, FILE_1)
    wait_until_status(id_1, JobStatus.RUNNING, TIMEOUT)

    time.sleep(20)

    client.stop_job(id_1)
    wait_until_status(id_1, {JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.FAILED}, TIMEOUT)

    # also we can do
    # client.delete_job(id_1)

    id_1 = run_in_cluster(3, FILE_1)
    wait_until_status(id_1, JobStatus.RUNNING, TIMEOUT)

    id_2 = run_in_cluster(1, FILE_2)
    wait_until_status(id_1, JobStatus.RUNNING, TIMEOUT)

    wait_until_status(id_1, {JobStatus.SUCCEEDED, JobStatus.FAILED}, TIMEOUT)
    wait_until_status(id_2, {JobStatus.SUCCEEDED, JobStatus.FAILED}, TIMEOUT)
    print("Succeeded to make models")





