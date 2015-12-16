<a name="top"></a>
TribeFlow
=========

1. [Top](#top)
2. [Datasets](#data)
3. [Competing Methods](#competition)

Contains the TribeFlow (previously node-sherlock) source code.

Dependencies
------------

The python dependencies are:

* Mpi4Py
* numpy
* scipy
* cython
* pandas
* plac

You will also need to install and setup: 

* OpenMP
* MPI

How to install dependencies
---------------------------

*Easy way:* Install [Anaconda Python](https://www.continuum.io/) and 
set it up as your default enviroment.

*Hard way:* Use pip or your package manager to install the dependencies. 

```bash
pip install numpy
pip install scipy
pip install cython
pip install pandas
pip install mpi4py
pip install plac
```

Use or package manager (*apt* on Ubuntu, *HomeBrew* on a mac) to install
OpenMP and MPI. These are the managers I tested with. Should work on any
other environment.

How to compile
--------------

Simply type `make`

```bash
make
```

How to use
----------

Either use `python setup.py install` to install the packager or just use it from
the package folder using the `run_script.sh` command.

*How to parse datasets:* Use the `scripts/trace_converter.py` script. It has a help.

For command line help:

```bash
$ python scripts/trace_converter.py -h
$ python main.py -h
```

Running with mpi

```bash
$ mpiexec -np 4 python main.py [OPTIONS]
```

Example
-------

The example below is the same code used for every result in the paper. It runs
TribeFlow with the options used in every result in the paper. Explaining the
parameters:

   * *-np 4* Number of cores for execution.
   * *100* topics.
   * *output.h5* model file.
   * *--kernel eccdf* The kernel heuristic for inter-event time estimation. ECCDF.
     based as per described on the paper. We also have a t-student kernel.
   * *--residency_priors 1 99* The priors for the inter-event time estimation.
   * *--leaveout 0.3* Number of transitions to leaveout.
   * *--num_iter 2000* Number of iterations.
   * *--num_batches 20* Number of split/merge moves.

```bash
$ mpiexec -np 4 python main.py trace.dat 100 output.h5 --kernel eccdf
--residency_priors 1 99 --leaveout 0.3 --num_iter 2000 --num_batches 20
```

<a name="data"></a>
Datasets
========

To come!

<a name="competition"></a>
Competing Methods
=================

* [PRLME](http://github.com/flaviovdf/plme)
* [FPMC](http://github.com/flaviovdf/fpmc)
* [LME](http://www.cs.cornell.edu/people/tj/playlists/index.html)
* [Gravity Model](https://github.com/flaviovdf/tribeflow/blob/master/scripts/gravity_model.py)
