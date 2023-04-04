# FFL - Fast Federated Learning

Fast Federated Learning (FFL) is a C/C++-based Federated Learning framework built on top of the parallel programming [FastFlow](http://calvados.di.unipi.it) framework. It exploits the [Cereal](https://uscilab.github.io/cereal/) library to efficiently serialise the updates sent over the network and the [libtorch](https://pytorch.org/cppdocs/installing.html) library to fully bypass the need for Python code.



## Getting started
---
To setup the whole system and to compile the example just run:
```
source setup.sh	
```
This script will automatically download all the required libraries, update the environment variables, build the `dff_run` utility, launch CMake and build all the available examples.



## Available examples
---
Four different examples are currently available, obtained as the combination of two different communication topologies (master-worker, peer-to-peer) and two execution modalities (shared-memory, distributed). The executables have the following names:
|        		| Shared-memory	| Distributed      |
|:------------- |:--------------| :----------------|
| Master-worker | `mnist` 		|  `mnist_dist`    |
| Peer-to-peer  | `mnist_p2p`	|  `mnist_p2p_dist`|

### Shared-memory examples
The shared-memory examples (`mnist`, `mnist_p2p`) are straght-forward to run. They can be executed with the following syntax:
```
./mnist [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_workers=3]
./mnist_p2p [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_peers=3]
```
where `forcecpu` indicates if to force the CPU use (1) or to allow the GPU use (0), `rounds` indicate the number of federated rounds to perform, `epochs/round` the number of training epoch to be run for each round, `data_path` is the path to the data files, and `num_workers`/`num_peers` is the number of clients to create.

### Distributed examples
The distributed examples require an additional file to run correctly: a `json` distributed configuration file specifying on which host each FastFlow instance will run and which is his role (especially for the master-worker scenario). A generic distrubted configuration file for `mnist_dist` looks like this:
```
{
    "groups" : [
     {   
        "endpoint" : "localhost:8000",
        "name" : "Federator"
     },
     {   
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
where endpoint indicates the host and port where the FastFlow instance will be created, and name specifies the role of the node:
| Name       | Meaning               | Use case                  |
|:-----------|:----------------------| :-------------------------|
| `Federator`| Server 		         | master-worker             |
| `W[N]`     | Client/Peer\[rank\]   | master-worker/peer-to-peer|

Once the distributed configuration file is available, then running the examples is simalar to the shared-memory scenario, but additionally requires the `dff_run` utility:
```
dff_run -V -p TCP -f [distributed_config_file] ./mnist_dist [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_workers=3]
dff_run -V -p TCP -f [distributed_config_file] ./mnist_p2p_dist [forcecpu=0/1] [rounds=10] [epochs/round=2] [data_path="../../data"] [num_peers=3]
```
where `-V` allow to visualise all the clients' output from the launching console, `-p TCP` forces the use of the TCP backend (MPI is also available), and `-f [distributed_config_file]` requires the distributed configuration file path.



## Required libraries
---
The whole project in wrote in C/C++ and to be compiled require a version of `CMake` > 3.0 and a C++20 compatible `C/C++ compiler`.

Furthermore, the following software libraries are needed:
| Library       | Version       | Link  														   |
|:------------- |:--------------| :----------------------------------------------------------------|
| `FastFlow`    | DistributedFF | [GitHub](https://github.com/fastflow/fastflow/tree/DistributedFF)|
| `Cereal`    	| 1.3.2		    | [GitHub](https://github.com/USCiLab/cereal/tree/v1.3.2)		   |
| `libtorch` 	| 2.0.0         | [Webpage](https://pytorch.org/get-started/locally/) 			   |



## Publications
---
This work has been published at the [2023 edition of the ACM Computing Frontiers conference](https://www.computingfrontiers.org/2023/).

The paper's citation and link will be provided as soon as they became available.



## Contacts

This repository is maintained by [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) (gianluca.mittone@unito.it).