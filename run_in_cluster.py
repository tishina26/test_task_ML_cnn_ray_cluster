# https://docs.ray.io/en/latest/cluster/kubernetes/examples/ml-example.html#submit-the-workload
import os
from os import fork
from ray.job_submission import JobSubmissionClient
import signal

LIMIT_CPUS = 4
CPUS_IN_USE = 0
# словарь файл - cpu
files_cpus = {}
# словарь ожидающего листа
waiting_list = {}

def run_in_cluster(CPUS, FILE_TO_RUN):
    client = JobSubmissionClient("http://127.0.0.1:8265")

    kick_off_xgboost_benchmark = (
        "git clone https://github.com/tishina26/test_task;"
        "python tishina26/test_task/" + FILE_TO_RUN
    )

    submission_id = client.submit_job(
        entrypoint=kick_off_xgboost_benchmark,
        entrypoint_num_cpus=CPUS
    )


def stop_model():
    # find model using max cpus
    max_cpus = 0
    file = ""
    for f, c in files_cpus.items():
        if (c > max_cpus):
            max_cpus = c
            file = f

    # stop this model, change cpus, continue with less cpus
    command = "pkill - f tishina26/test_task/" + file
    os.system(command)

    max_cpus -= 1
    CPUS_IN_USE -= 1
    files_cpus[file] = max_cpus
    # модель сохранялась, поэтому сможет загрузить последнее сохранение и продолжить обучение
    run_in_cluster(max_cpus, file)

if __name__ == "__main__":
    # установим реакцию на сигнал
    # signal.signal(signal.SIGINT, handler_sigint)
    if (fork()):
        while (True):
            if (len(waiting_list) > 0 and CPUS_IN_USE > 0):
                # можно запустить какой-то файл
                run_in_cluster(min(CPUS_IN_USE, waiting_list[0]), waiting_list[0])
    else:
        while (True):
            args = input("введите команду ").split(" ")
            if (args[0] == "run" and args[2] == "with"):
                if (LIMIT_CPUS - CPUS_IN_USE > 0):
                    if (args[4] == "and" and int(args[5]) <= LIMIT_CPUS - CPUS_IN_USE):
                        files_cpus[args[1]] = int(args[5])
                        CPUS_IN_USE += int(args[5])
                        run_in_cluster(int(args[5]), args[1])
                    elif (args[4] == "and" and int(args[5]) > LIMIT_CPUS - CPUS_IN_USE):
                        print("cpus available: " + str(LIMIT_CPUS - CPUS_IN_USE))
                        files_cpus[args[1]] = LIMIT_CPUS - CPUS_IN_USE
                        CPUS_IN_USE += LIMIT_CPUS - CPUS_IN_USE
                        run_in_cluster(LIMIT_CPUS - CPUS_IN_USE, args[1])
                    else:
                        print("1 cpu will be taken")
                        files_cpus[args[1]] = 1
                        CPUS_IN_USE += 1
                        run_in_cluster(1, args[1])

                else:
                    print("no cpus available")
                    if (int(args[3]) == 0):
                        print("1 cpu will be free")
                        stop_model()
                        files_cpus[args[1]] = 1
                        CPUS_IN_USE += 1
                        run_in_cluster(1, args[1])
                    else:
                        print("please, wait until any cpu will be available")
                        waiting_list[args[1]] = int(args[5])

            else:
                print('template = run "file" with "priority" and "cpus"')
                print('default "cpus" = 1')

