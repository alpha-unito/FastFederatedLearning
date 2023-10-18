"""
FastFlow Sequential and Parallel code wrapper
"""
from typing import Final, List, TextIO

from .building_block import BuildingBlock

worker_creation_ms: Final[str] = """
        ff_node *worker = new ff_comb(new MiNodeAdapter<StateDict>,
                                      new Worker(i, net, net->state_dict(), train_epochs, optimizer,
                                                 torch::data::make_data_loader(train_dataset,
                                                                               torch::data::samplers::DistributedRandomSampler(
                                                                                       train_dataset.size().value(),
                                                                                       num_workers,
                                                                                       i % num_workers,
                                                                                       true),
                                                                               train_batchsize),
                                                 torch::data::make_data_loader(test_dataset, test_batchsize),
                                                 device));
        w.push_back(worker);
        a2a.createGroup("W" + std::to_string(i)) << worker;
    """

worker_creation_p2p: Final[str] = """
        ff_node *peer = new ff_comb(new MiNodeAdapter<StateDict>(),
                                            new Peer(i, local_net, local_net->state_dict(), train_epochs, optimizer,
                                                     torch::data::make_data_loader(train_dataset,
                                                                                   torch::data::samplers::DistributedRandomSampler(
                                                                                           train_dataset.size().value(),
                                                                                           num_workers,
                                                                                           i % num_workers,
                                                                                           true),
                                                                                   train_batchsize),
                                                     fed_net, fed_net->state_dict(), aggregator, num_workers, rounds,
                                                     torch::data::make_data_loader(test_dataset, test_batchsize),
                                                     device), true, true);
        left.push_back(peer);
    """


class Wrapper(BuildingBlock):
    """ FastFlow Sequential and Parallel code wrapper """

    def __init__(self, label: str):
        """ Initialisation of a Wrapper Building Block.

        :param label: label assigned to the Wrapper; supported values: Train, Test
        :type label: string
        """
        super().__init__(self.__str__())

        self.label: Final[str] = label

    def get_label(self) -> str:
        """ Getter for the Wrapper Building Block label.

        :return: wrapper's label
        :rtype: str
        """
        return self.label

    def compile(self, building_blocks: List[BuildingBlock], source_file: TextIO):
        """ Compilation of the Wrapper Building Block

        :param building_blocks: remaining Building Blocks to compile.
        :type building_blocks: List[BuildingBlock]
        :param source_file: C/C++ source file to write on.
        :type source_file: TextIO
        """
        if building_blocks:
            if self.label == "Train":
                first_bb: BuildingBlock
                remaining_bb: List[BuildingBlock]
                first_bb, *remaining_bb = building_blocks
                if isinstance(first_bb, Wrapper) and first_bb.get_label() == "Test":
                    if remaining_bb:
                        source_file.write(worker_creation_p2p)
                    else:
                        source_file.write(worker_creation_ms)
                self.logger.debug("Analysing the %s task...", first_bb)
                first_bb.compile(remaining_bb, source_file)
            elif self.label == "Test":
                first_bb: BuildingBlock
                remaining_bb: List[BuildingBlock]
                first_bb, *remaining_bb = building_blocks
                self.logger.debug("Analysing the %s task...", first_bb)
                first_bb.compile(remaining_bb, source_file)
