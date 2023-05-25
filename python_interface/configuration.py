from custom_types import PathLike


class Configuration(dict):

    def __init__(self, json_path: PathLike, data_path: PathLike, runner_path: PathLike, executable_path: PathLike):
        super().__init__()

        self["json_path"] = json_path
        self["data_path"] = data_path
        self["runner_path"] = runner_path
        self["executable_path"] = executable_path

    def get_json_path(self) -> PathLike:
        return self["json_path"]

    def get_data_path(self) -> PathLike:
        return self["data_path"]

    def get_runner_path(self) -> PathLike:
        return self["runner_path"]

    def get_executable_path(self) -> PathLike:
        return self["executable_path"]

    def set_json_path(self, json_path: PathLike):
        self["json_path"] = json_path

    def set_data_path(self, data_path: PathLike):
        self["data_path"] = data_path

    def set_runner_path(self, runner_path: PathLike):
        self["runner_path"] = runner_path

    def set_executable_path(self, executable_path: PathLike):
        self["executable_path"] = executable_path
