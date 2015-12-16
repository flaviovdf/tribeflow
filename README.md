TribeFlow
---------

Contains the TribeFlow (previously node-sherlock) source code.

Dependencies
============

* numpy
* scipy
* cython
* pandas
* plac

How to install dependencies
===========================

*Easy way:* Install anaconda python and set it up as your default enviroment

*Hard way:* Use pip or your package manager to install the dependencies.


```bash
pip install numpy
pip install scipy
pip install cython
pip install pandas
pip install plac
```

How to use
==========

Either use `python setup.py install` to install the packager or just use it from
the package folder using the `run_script.sh` command.

*How to parse datasets:* Use the `scripts/trace_converter.py` script. It has a help.

Example (for command line help):

```bash
./run_script.sh scripts/trace_converter.py -h
./run_script.sh main.py -h
```
