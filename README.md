<a name="top"></a>
TribeFlow
=========

1. [Home](#top)
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

**Converting the Trace**

Let's we have a trace like the Last.FM trace from [Oscar
Celma](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html).
In this example, each line is of the form:

```bash
userid \t timestamp \t musicbrainz-artist-id \t artist-name \t
musicbrainz-track-id \t track-name
```

For instance:

```bash
user_000001 2009-05-01T09:17:36Z    c74ee320-1daa-43e6-89ee-f71070ee9e8f
Impossible Beings   952f360d-d678-40b2-8a64-18b4fa4c5f8Dois Pólos
```

First, we want to convert this file to our input format. We do this with the
`scripts/trace_converter.py` script. Let's have a look at the options from
this script:

```bash
$ python scripts/trace_converter.py -h
usage: trace_converter.py [-h] [-d DELIMITER] [-l LOOPS] [-r SORT] [-f FMT]
                          [-s SCALE] [-k SKIP_HEADER] [-m MEM_SIZE]
                          original_trace tstamp_column hypernode_column
                          obj_node_column

positional arguments:
  original_trace        The name of the original trace
  tstamp_column         The column of the time stamp
  hypernode_column      The column of the time hypernode
  obj_node_column       The column of the object node

optional arguments:
  -h, --help            show this help message and exit
  -d DELIMITER, --delimiter DELIMITER
                        The delimiter
  -l LOOPS, --loops LOOPS
                        Consider loops
  -r SORT, --sort SORT  Sort the trace
  -f FMT, --fmt FMT     The format of the date in the trace
  -s SCALE, --scale SCALE
                        Scale the time by this value
  -k SKIP_HEADER, --skip_header SKIP_HEADER
                        Skip these first k lines
  -m MEM_SIZE, --mem_size MEM_SIZE
                        Memory Size (the markov order is m - 1)
```

The positional (obrigatory) arguments are:

   * *original_trace* is the input file
   * *hypernode_column* represents the users (called hypernodes since it can 
     be playlists as well)
   * *tstamp_column* the column of the time stamp
   * *obj_node_column* the objects of interest

We can convert the file with the following line:

```bash
python scripts/trace_converter.py scripts/test_parser.dat 1 0 2 -d$'\t' \
        -f'%Y-%m-%dT%H:%M:%SZ' > trace.dat
```

Here, we are saying that column 1 are the timestamps, 0 is the user, and 2 are the
objects (artist ids). The delimiter *-d* is a tab. The time stamp format is
`'%Y-%m-%dT%H:%M:%SZ'`.

**Learning the Model**

The example below is the same code used for every result in the paper. It runs
TribeFlow with the options used in every result in the paper. Explaining the
parameters:

   * *-np 4* Number of cores for execution.
   * *100* topics.
   * *output.h5* model file.
   * *--kernel eccdf* The kernel heuristic for inter-event time estimation. ECCDF
     based as per described on the paper. We also have a t-student kernel.
   * *--residency_priors 1 99* The priors for the inter-event time estimation.
   * *--leaveout 0.3* Number of transitions to leaveout.
   * *--num_iter 2000* Number of iterations.
   * *--num_batches 20* Number of split/merge moves.

*The example below uses 20 cores*
```bash
$ mpiexec -np 20 python main.py trace.dat 100 output.h5 \
    --kernel eccdf --residency_priors 1 99 \
    --leaveout 0.3 --num_iter 2000 --num_batches 20
```

<a name="data"></a>
Datasets
========

Below we have the list of datasets explored on the paper. We also curated links
to various other timestamp datasets that can be exploited by TribeFlow and 
future efforts.

Datasets used on the paper:

1. [LastFM-1k](https://archive.org/details/201309_foursquare_dataset_umn)
2. *LastFM-Our* Drop me an e-mail for now, looking for a place to upload it.
3. [FourSQ](https://archive.org/details/201309_foursquare_dataset_umn)
    This dataset was removed from the original website. Still available on
    archive. Other, more recent, FourSQ datasets are available. See below.
4. [Brightkite](https://snap.stanford.edu/data/loc-brightkite.html)
6. [Yes](http://www.cs.cornell.edu/people/tj/playlists/index.html)

List of other, some more recent, datasets that can be explored by TribeFlow.

1. [Newer FourSQ](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
2. [Million Music Tweet](http://www.cp.jku.at/datasets/MMTD/)
3. [Movie Ratings](https://github.com/sidooms/MovieTweetings)
4. [Twitter](https://snap.stanford.edu/data/twitter7.html)
5. [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
6. [Yelp](https://www.yelp.com/dataset_challenge)
7. [Best Buy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data)

Basically, anything with users (playlists, actors, etc also work), objects and 
timestamps.

<a name="competition"></a>
Competing Methods
=================

* [PRLME](http://github.com/flaviovdf/plme)
* [FPMC](http://github.com/flaviovdf/fpmc)
* [LME](http://www.cs.cornell.edu/people/tj/playlists/index.html)
* [Gravity Model](https://github.com/flaviovdf/tribeflow/blob/master/scripts/gravity_model.py)
* [TMLDA](https://github.com/flaviovdf/tribeflow/blob/master/scripts/tmlda.py)
