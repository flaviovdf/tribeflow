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
./run_script.sh scripts/trace_converter.py -h
./run_script.sh main.py -h
```

Running with mpi

```bash
mpiexec -np 4 python main.py [OPTIONS]
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
