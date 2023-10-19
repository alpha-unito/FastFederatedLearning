"""
This class handles the creation of the json configuration file necessary for building a federation with FastFlow
"""
import json
import logging
from typing import List, Dict, Union, Optional, NoReturn

from python_interface.custom.custom_types import PathLike
from python_interface.utils import constants
from python_interface.utils.constants import Topology
from python_interface.utils.utils import check_var_in_literal, get_logger, check_and_create_path


# TODO: This class is currently static, meaning that no new nodes can be addedd to the configuration
# TODO: Implements checks for length when using setters

class JSONGenerator(dict):
    """Creation of a FFjson JSON object containing all the necessary information for a FastFlow execution."""

    def __init__(self, topology: Topology = constants.MASTER_WORKER,
                 endpoints: Union[int, List[str], List[Dict[str, str]]] = 2,
                 commands: Optional[Union[str, List[str]]] = None):
        """Simple class for creating and modelling JSON file formatted for working with FastFlow.
        This class directly extends the base 'dict' class from Python, so can be treated as a plain dictionary
        with additional features.

        :param topology: Typology of communication topology to create.
            Actual choices are constant.MASTER_WORKER and constant.PEER_TO_PEER.
        :type topology: Topology
        :param endpoints: Specification of the endpoints (nodes, both clients and server). Can be specified in many ways:
            - number of clients, by default will all be allocated on the localhost (useful for testing);
            - list of [ip or hostname]:port of the different nodes (in the master-worker case the first node is assumed
                as server);
            - list of already formatted entries to be directly injected in the FFjson object.
        :type endpoints: Union[int, List[str], List[Dict[str, str]]]
        :param commands: Specific commands to be executed on the device host before executing the FastFlow processes.
            Can be specified in many ways:
            - single command-line command as str to be assigned to all endpoints;
            - as a list of string, each one to be paired to each device in order.
        :type commands: Optional[Union[str, List[str]]]
        """
        super().__init__()
        self.logger: logging.Logger = get_logger(self.__class__.__name__)

        self[constants.GROUPS]: List[Dict]
        self.set_endpoints(endpoints)
        self.set_topology(topology)
        self.set_commands(commands)

        self.logger.info("FastFlow configuration parameters correctly created.")

    def get_endpoints(self) -> List[str]:
        """Getter for the endpoints addresses.

        :return: List of ip(or hostname):port.
        :rtype: List[str]
        """
        return [entry[constants.ENDPOINT] for entry in self[constants.GROUPS]]

    def get_hosts(self) -> List[str]:
        """Getter for the endpoints addresses.

        :return: List of ip(or hostname).
        :rtype: List[str]
        """
        return [entry[constants.ENDPOINT].split(":")[0] for entry in self[constants.GROUPS]]

    def get_names(self) -> List[str]:
        """Getter for the endpoints names.

        :return: List of names assigned to each endpoint.
        :rtype: List[str]
        """
        return [entry[constants.NAME] for entry in self[constants.GROUPS]]

    def get_commands(self) -> List[str]:
        """Getter for the commands associated to each device.

        :return: List of commands assigned to each endpoint.
        :rtype: List[str]
        """
        return [entry[constants.PRE_CMD] for entry in self[constants.GROUPS]]

    def get_json(self) -> str:
        """Getter for the JSON-formatted version of the FFjson object.

        :return: JSON-formatted version of the FFjson object.
        :rtype: str
        """
        return json.dumps(self, indent='\t', sort_keys=True)

    def get_clients_number(self) -> int:
        """Return the number of clients involved in the federation.

        :return: Number of clients involved in the federation.
        :rtype: int
        """
        clients: Optional[int] = None
        match self["topology"]:
            case constants.MASTER_WORKER | constants.EDGE_INFERENCE:
                clients = len(self[constants.GROUPS]) - 1
            case constants.PEER_TO_PEER | constants.CUSTOM:
                clients = len(self[constants.GROUPS])
            case constants.MVDET:
                clients = len(self[constants.GROUPS]) - 3

        return clients

    def set_endpoints(self, endpoints: Union[int, List[str], List[Dict[str, str]]]) -> NoReturn | ValueError:
        """Creation of the nodes' description.

        :param endpoints: Specification of the endpoints (nodes, both clients and server). Can be specified in many ways:
            - number of clients, by default will all be allocated on the localhost (useful for testing);
            - list of [ip or hostname]:port of the different nodes (in the master-worker case the first node is assumed
                as server);
            - list of already formatted entries to be directly injected in the FFjson object.
        :type endpoints: int or List[str] or List[Dict[str, str]]
        """
        if isinstance(endpoints, int):
            self[constants.GROUPS] = [{constants.ENDPOINT: constants.DEFAULT_ENDPOINT} for _ in range(endpoints)]
            self.logger.debug("Created endpoints: %s", self[constants.GROUPS])
        elif isinstance(endpoints, list):
            if isinstance(endpoints[0], str):
                self[constants.GROUPS] = [{constants.ENDPOINT: host} for host in endpoints]
            elif isinstance(endpoints[0], dict):
                self[constants.GROUPS] = endpoints
            self.logger.debug("Created endpoints: %s", self[constants.GROUPS])
        else:
            self.logger.critical("Specified endpoints format not supported: %s", endpoints)
            raise ValueError("Specified endpoints format not supported: " + str(endpoints))

    def set_topology(self, topology: Topology) -> NoReturn | ValueError:
        """Setter for the Topology type.

        :param topology: chosen topology.
        :type topology: Topology
        """
        try:
            check_var_in_literal(topology, Topology, self.logger)
        except ValueError as e:
            self.logger.critical("Specified topology is not supported: %s", e)
            raise e
        self.logger.info("Setting the FastFlow topology to %s", topology)
        self["topology"] = topology

        self.logger.info("Creating FastFlow names for topology %s", self["topology"])
        match self["topology"]:
            case constants.MASTER_WORKER | constants.PEER_TO_PEER | constants.EDGE_INFERENCE | constants.CUSTOM:
                counter: int = 0
                for entry in self[constants.GROUPS]:
                    entry[constants.NAME] = constants.WORKER(counter)
                    counter += 1
            case constants.MVDET:
                counter: int = 0
                aggregator_needed: bool = False
                for entry in self[constants.GROUPS]:
                    if counter == 0:
                        entry[constants.NAME] = constants.SOURCE(counter)
                    elif 1 <= counter < 8 and not aggregator_needed:
                        entry[constants.NAME] = constants.WORKER(counter)
                        if counter % 7 == 0:
                            aggregator_needed: bool = True
                            continue
                    elif aggregator_needed:
                        entry[constants.NAME] = constants.AGGREGATOR(int(counter / 7))
                        aggregator_needed = False
                    elif counter == 8:
                        entry[constants.NAME] = constants.SOURCE(counter)
                    counter += 1
            case _:
                self.logger.critical("Topology type not supported: %s", self["topology"])
                raise ValueError("Topology type not supported: " + self["topology"])
        self.logger.debug("Created FastFlow names: %s", self[constants.GROUPS])

    def set_commands(self, commands: Optional[Union[str, List[str]]] = None) -> NoReturn | ValueError:
        """Association of the command-line commands to each device.

        :param commands: Specific commands to be executed on the device host before executing the FastFlow processes.
            Can be specified in many ways:
            - single command-line command as str to be assigned to all endpoints;
            - as a list of string, each one to be paired to each device in order.
        :type commands: Optional[Union[str, List[str]]]
        """
        if commands is not None:
            if isinstance(commands, str):
                for entry in self[constants.GROUPS]:
                    entry[constants.PRE_CMD] = commands
            elif isinstance(commands, list):
                for entry, cmd in zip(self[constants.GROUPS], commands):
                    entry[constants.PRE_CMD] = cmd
            else:
                self.logger.critical("Specified pre-command format not supported: %s", commands)
                raise ValueError("Specified endpoints format not supported: " + str(commands))
        self.logger.debug("Created pre commands: %s", self[constants.GROUPS])

    def generate_json_file(self, path: PathLike):
        """Save the configuration to an actual .json file.

        :param path: The path where to save the .json file
        :type path: PathLike
        """
        check_and_create_path(path, "JSON FastFlow configuration file", self.logger)
        with open(path, "w") as text_file:
            text_file.write(self.get_json())
        self.logger.info("JSON configuration file saved to %s", path)
