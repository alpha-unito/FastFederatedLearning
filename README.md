# FFL - Fast Federated Learning

Fast Federated Learning (FFL) is a C/C++-based Federated Learning framework built on top of the parallel programming [FastFlow](http://calvados.di.unipi.it) framework. It exploits the [Cereal](https://uscilab.github.io/cereal/) library to efficiently serialise the updates sent over the network and the [libtorch](https://pytorch.org/cppdocs/installing.html) library to fully bypass the need for Python code. It has been successfully tested on x86_64, ARM and RISC-V platforms. FFL comes with scripts for automatically installing the framework and reproducing all the experiments reported in the original paper.



## Getting started
To setup the whole system and to compile the example just run:
```
source setup.sh	
```
This script will automatically download all the required libraries, update the environment variables, build the `dff_run` utility, launch CMake and build all the available examples.


## Reproducibility

Starting from a fresh Ubuntu 22.04 installation, the first step to reproduce the results reported in the paper is to update the available packages information and install the `build-essential`, `cmake`, `libopencv-dev`, and `unzip` packages:
```
sudo apt-get update
sudo apt-get install build-essential cmake libopencv-dev unzip
```
Then, clone the FFL repository locally and build the code as follows (the `setup.sh` script will take care of everything, as mentioned in the previous section):
```
git clone https://github.com/alpha-unito/FastFederatedLearning.git
cd FastFederatedLearning
source setup.sh
```
Lasty, all that is needed to run the full set of experiments is to run the `reproduce.sh` script:
```
bash reproduce.sh
```
This script will take care of running in a replicate manner (5 times) all available examples (3) in all the available configurations (3), for a total of 5\*3\*3=45 runs. The mean execution time for each combination will be reported on the output, and logs will be saved for each experiment.
Inside the reproduce script, the `MAX_ITER` variable can be set to change the replica factor of the experiments.

### Power measurements

The power measurements on the x86_64 and ARM platforma has been done trhough the [Powermon](https://github.com/Yamagi/powermon) utility. This software has to be started in parallel with the code execution on another shell, and then killed right after the desired computation is concluded.

On the RISC-V platform, on the other hand, we used a National Instruments oscilloscope directly connected to the hardware board. This has been done due to the lack of other software-based solution and the high precision of the available machinery.



## Available examples
Six different examples are currently available, obtained as the combination of three different communication topologies (master-worker, peer-to-peer, tree-based) and two execution modalities (shared-memory, distributed). The executables have the following names:
| Topology   	 | Shared-memory	 | Distributed            |
|:------------- |:--------------- | :--------------------- |
| Master-worker | `masterworker`  |  `masterworker_dist`   |
| Peer-to-peer  | `p2p`	          |  `p2p_dist`            | 
| Tree-based    | `edgeinference` |  `edgeinference_dist`  |

### Shared-memory examples
The shared-memory examples (`masterworker`, `p2p`, and `edgeinference`) are straght-forward to run. They can be executed with the following syntax:
```
./masterworker     [forcecpu=0/1] [rounds=10] [epochs/round=2]   [data_path="../../data"] [num_workers=3]
./p2p [forcecpu=0/1] [rounds=10] [epochs/round=2]   [data_path="../../data"] [num_peers=3]
./edgeinference   [forcecpu=0/1] [groups=3]  [clients/groups=1] [model_path] [data_path]
```
where `forcecpu` indicates if to force the CPU use (1) or to allow the GPU use (0), `rounds` indicate the number of federated rounds to perform, `epochs/round` the number of training epoch to be run for each round, `data_path` is the path to the data files, `num_workers`/`num_peers`/`groups` is the number of instances to create, `client/groups` is the number of clients in each group, and `model_path` is the path of the `torchscript` model to be used.

### Distributed examples
The distributed examples require an additional file to run correctly: a `json` distributed configuration file specifying on which host each FastFlow instance will run and which is his role (especially for the master-worker scenario). A generic distrubted configuration file for `masterworker_dist` looks like this:
```
{
   "groups" : [
      {   
         "preCmd" : "MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 taskset -c 0-3",
         "endpoint" : "localhost:8000",
         "name" : "Federator"
      },
      {   
         "preCmd" : "MKL_NUM_THREADS=4 OMP_NUM_THREADS=4 taskset -c 20-23",
         "endpoint" : "128.0.0.1:8001",
         "name" : "W0"
      },
      {   
         "endpoint" : "host2:8022",
         "name" : "W1"
      },
      {   
         "endpoint" : "134.342.12.12:6004",
         "name" : "W2"
      }
   ]
}
```
where preCmd indicates commands to be appended before the training command, endpoint indicates the host and port where the FastFlow instance will be created, and name specifies the role of the node:
| Name       | Meaning               | Use case                  |
|:-----------|:----------------------| :-------------------------|
| `Federator`| Server 		          | master-worker             |
| `W[N]`     | Client/Peer\[rank\]   | master-worker/peer-to-peer|
| `G[N]`     | Group\[rank\] (0 means root) | tree-based         |

Once the distributed configuration file is available, then running the examples is simalar to the shared-memory scenario, but additionally requires the `dff_run` utility:
```
dff_run -V -p TCP -f [distributed_config_file] ./masterworker_dist [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_workers=3]
dff_run -V -p TCP -f [distributed_config_file] ./p2p_dist [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_peers=3]
dff_run -V -p TCP -f [distributed_config_file] ./edgeinference [forcecpu=0/1] [groups=3] [clients/groups=1] [model_path] [data_path]
```
where `-V` allow to visualise all the clients' output from the launching console, `-p TCP` forces the use of the TCP backend (MPI is also available), and `-f [distributed_config_file]` requires the distributed configuration file path.



## Required libraries
The whole project in wrote in C/C++ and to be compiled require a version of `CMake` > 3.0 and a C++17 compatible `C/C++ compiler`.

Furthermore, the following software libraries are needed:
| Library       | Version       | Link  														                  |
|:------------- |:--------------| :----------------------------------------------------------------|
| `FastFlow`    | DistributedFF | [GitHub](https://github.com/fastflow/fastflow/tree/DistributedFF)|
| `Cereal`    	 | 1.3.2		     | [GitHub](https://github.com/USCiLab/cereal/tree/v1.3.2)		      |
| `libtorch` 	 | 2.0.0         | [Webpage](https://pytorch.org/get-started/locally/) 			      |
| `OpenCV` 	    | 4.6.0         | [Webpage](https://opencv.org) 			      |


## Publications
This work has been published at the [2023 edition of the ACM Computing Frontiers conference](https://www.computingfrontiers.org/2023/).

The paper's citation and link will be provided as soon as they became available.



## Contacts
This repository is maintained by [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) (gianluca.mittone@unito.it).