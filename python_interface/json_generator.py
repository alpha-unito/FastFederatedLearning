"""
This class handles the creation of the json configuration files necessary for executing a federation with FastFlow
"""

from typing import List, Dict, Union, Optional
import constants
import json


# @TODO: This class is currently static, meaning that no new nodes can be addedd to the configuration
# @TODO: Implements checks for length when using setters

class FFjson(dict):
    """Simple class for creating and modelling JSON file formatted for working with FastFlow.
        This class directly extends the base dict class from Python, so can be threated as a plain dictionary with
        additional features.
    """

    def __init__(self, topology: str = constants.MASTER_WORKER,
                 endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None):
        """Creation of a FFjson JSON object containing all the necessary information for a FastFlow execution.

        :param
        topology : str
            Typology of communication topology to create.
            Actual choices are constant.MASTER_WORKER and constant.PEER_TO_PEER.
        endpoints : Union[int, List[str], List[Dict[str, str]]]
            Specification of the endpoints (nodes, both clients and server). Can be specified in many ways:
            - number of clients, by default will all be allocated on the localhost (useful for testing);
            - list of [ip or hostname]:port of the different nodes (in the master-worker case the first node is assumed
                as server);
            - list of already formatted entries to be directly injected in the FFjson object.
        commands : Optional[Union[str, List[str]]]
            Specific commands to be executed on the device host before executing the FastFlow processes.
            Can be specified in many ways:
            - single command-line command as str to be assigned to all endpoints;
            - as a list of string, each one to be paired to each device in order.
        """
        super().__init__()

        if topology not in constants.TOPOLOGIES:
            raise ValueError("Topology type not supported: " + str(topology))

        self.topology: str = topology
        self[constants.GROUPS]: List[Dict]

        self.create_endpoints(endpoints)
        self.create_commands(commands)
        self.create_names(topology)

    def create_endpoints(self, endpoints: Union[int, List[str], List[Dict[str, str]]]):
        """Creation of the nodes' description.

        :param
        endpoints: int or List[str] or List[Dict[str, str]]
            Specification of the endpoints (nodes, both clients and server). Can be specified in many ways:
            - number of clients, by default will all be allocated on the localhost (useful for testing);
            - list of [ip or hostname]:port of the different nodes (in the master-worker case the first node is assumed
                as server);
            - list of already formatted entries to be directly injected in the FFjson object.
        """
        if isinstance(endpoints, int):
            self[constants.GROUPS] = [{constants.ENDPOINT: constants.DEFAULT_ENDPOINT} for _ in range(endpoints)]
        elif isinstance(endpoints, list):
            if isinstance(endpoints[0], str):
                self[constants.GROUPS] = [{constants.ENDPOINT: host} for host in endpoints]
            elif isinstance(endpoints[0], dict):
                self[constants.GROUPS] = endpoints

    def create_names(self, topology: str = constants.MASTER_WORKER):
        """Creation of the nodes' names according to FastFow policy.

        :param
        topology : str
            Typology of communication topology to create.
            Actual choices are constant.MASTER_WORKER and constant.PEER_TO_PEER.
        """
        if topology == constants.MASTER_WORKER:
            counter: int = -1
            for entry in self[constants.GROUPS]:
                if counter == -1:
                    entry[constants.NAME] = constants.FEDERATOR
                else:
                    entry[constants.NAME] = constants.WORKER(counter)
                counter += 1
        elif topology == constants.PEER_TO_PEER:
            counter: int = 0
            for entry in self[constants.GROUPS]:
                entry[constants.NAME] = constants.WORKER(counter)
                counter += 1
        else:
            raise ValueError("Topology type not supported: " + str(topology))

    def create_commands(self, commands: Optional[Union[str, List[str]]] = None):
        """Association of the command-line commands to each device.

        :param
        commands : Optional[Union[str, List[str]]]
            Specific commands to be executed on the device host before executing the FastFlow processes.
            Can be specified in many ways:
            - single command-line command as str to be assigned to all endpoints;
            - as a list of string, each one to be paired to each device in order.
        """
        if commands is not None:
            if isinstance(commands, str):
                for entry in self[constants.GROUPS]:
                    entry[constants.PRE_CMD] = commands
            elif isinstance(commands, list):
                for entry, cmd in zip(self[constants.GROUPS], commands):
                    entry[constants.PRE_CMD] = cmd

    def get_endpoints(self) -> List[str]:
        """Getter for the endpoints addresses.

        :return: List[str]
            List of ip(or hostname):port.
        """
        return [entry[constants.ENDPOINT] for entry in self[constants.GROUPS]]

    def get_names(self) -> List[str]:
        """Getter for the endpoints names.

        :return: List[str]
            List of names assigned to each endpoint.
        """
        return [entry[constants.NAME] for entry in self[constants.GROUPS]]

    def get_commands(self) -> List[str]:
        """Getter for the commands associated to each device.

        :return: List[str]
            List of commands assigned to each endpoint.
        """
        return [entry[constants.PRE_CMD] for entry in self[constants.GROUPS]]

    def get_json(self) -> str:
        """Getter for the JSON-formatted version of the FFjson object.

        :return: str
            JSON-formatted version of the FFjson object.
        """
        return json.dumps(self, indent='\t', sort_keys=True)

    def set_endpoints(self, endpoints: Union[str, List[str]]):
        """Setter for the endpoints addresses and ports.

        :param
        endpoints: str or List[str]
            Specification of the endpoints (nodes, both clients and server). Can be specified in many ways:
            - single [ip or hostname]:port to be associated to all devices;
            - list of [ip or hostname]:port of the different nodes (in the master-worker case the first node is assumed
                as server);
        """
        if isinstance(endpoints, str):
            for entry in self[constants.GROUPS]:
                entry[constants.ENDPOINT] = endpoints
        elif isinstance(endpoints, list):
            for entry, endpoint in zip(self[constants.GROUPS], endpoints):
                entry[constants.ENDPOINT] = endpoint

    def set_names(self, names: List[str]):
        """Setter for the device names (should follow FastFlow conventions).

        :param
        names : List[str]
            List of names to be associated to each device.
        """
        for entry, name in zip(self[constants.GROUPS], names):
            entry[constants.NAME] = name

    def set_commands(self, commands: Union[str, List[str]]):
        """Setter for the commands to be executed before the FastFlow processes are started.

        :param
        commands: str or List[str]
            Command-line command. Can be specified in many ways:
            - single command str to be associated to all devices;
            - list of str commands for the different nodes.
        """
        if isinstance(commands, str):
            for entry in self[constants.PRE_CMD]:
                entry[constants.PRE_CMD] = commands
        elif isinstance(commands, list):
            for entry, command in zip(self[constants.PRE_CMD], commands):
                entry[constants.PRE_CMD] = command

    def generate_json_file(self, path: str):
        with open(path, "w") as text_file:
            text_file.write(self.get_json())

    def get_clients_number(self) -> int:
        return len(self[constants.GROUPS]) - 1 if self.topology == constants.MASTER_WORKER else len(
            self[constants.GROUPS])


if __name__ == '__main__':
    json_file = FFjson(endpoints=["localhost:8000", "localhost:8001", "localhost:8002", "localhost:8003"],
                       commands="MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 taskset -c 0-3", )
    print(json_file.get_json())
    quit()
