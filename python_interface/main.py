import constants
from json_generator import FFjson
from configuration import Configuration
from subprocess import call

CONFIG_PATH = "/mnt/shared/mittone/FastFederatedLearning/workspace/config.json"
EXECUTABLE_PATH_MS = "/mnt/shared/mittone/FastFederatedLearning/build/examples/masterworker/masterworker_dist"
DFF_RUN_PATH = "/mnt/shared/mittone/FastFederatedLearning/libs/fastflow/ff/distributed/loader/dff_run"
DATA_PATH = "/mnt/shared/mittone/FastFederatedLearning/data"

json = FFjson(endpoints=["device" + str(rank) + ":800" + str(rank) for rank in range(1, 21)],
              topology=constants.MASTER_WORKER)

config = Configuration(json_path=CONFIG_PATH, data_path=DATA_PATH, runner_path=DFF_RUN_PATH,
                       executable_path=EXECUTABLE_PATH_MS)


def run_experiment(config: Configuration, json: FFjson):
    json.generate_json_file(CONFIG_PATH)
    call([config.get_runner_path(), "-V", "-p", "TCP", "-f ", config.get_json_path(), config.get_executable_path(), "1",
          "1", "1", config.get_data_path(), str(json.get_clients_number())])


run_experiment(config, json)
