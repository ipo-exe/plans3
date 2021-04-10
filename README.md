# `plans3` - Planning Nature-based Solutions

## What is `plans3`?
**Planning Nature-based Solutions** (`plans3`) is a modelling framework for planning the 
expansion of nature-based solutions in watersheds.

Warning! 

This is an underdevelopment project!

### Why the "3" on `plans3`?
`plans` was born in 2018 within the scope of a master's degree research project. 
While `plans1` was just a handful of python scripts, `plans2` has an application-like structure. 
Now, `plans3` has deep changes in hydrology modelling.

## What is included in this repository

- [x] All files required to run `plans3`;
- [x] A directory called `samples` with examples of input files;
- [x] A markdown file called `iofiles.md` of I/O file documentation;
- [ ] A markdown file called `guide.md` for a quick guide of `plans3` applications;
- [ ] A comprehensive `plans3_handbook.pdf` document.

## `python` and packages required

`plans3` is built on top of:
* [python 3.8](https://www.python.org/downloads/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scipy](https://www.scipy.org/)

## How to install `plans3` on a local machine

### Step 1: install python 3.8+
Go to https://www.python.org/downloads/ and download it. Make sure you add Python to PATH (checkbox on the installation wizard).

### Step 2: install the packages
To run `plans3` you need `numpy`, `pandas`, `matplotlib` and `scipy`. If you do not have it already installed, you can do that by using terminal or an IDE, like PyCharm.

On Windows terminal (you may find on the web similar installation procedures for Mac and Ubuntu):

`C:\Windows\System32\python -m pip install --upgrade pip` (this will update `pip`)

then:
`C:\Windows\System32\python -m pip install numpy`

then:
`C:\Windows\System32\python -m pip install pandas`

then:
`C:\Windows\System32\python -m pip install matplotlib`

and then:
`C:\Windows\System32\python -m pip install scipy`

### Step 3: download a clone of this repository
Download the ZIP file for the entire repository. Extract the files on any diretory of your machine.

## Running `plans3` by the Terminal User Interface (`TUI`)
After installing `plans3` on your machine, double-click on `run_tui.py` and it will run the terminal-based application. 

Alternatively, you may create a python file on the same directory, 
write the following code and then execute it:
```python
from tui import main  # this imports the main() function from module tui.py

main()  # call the main() function
```
A terminal-based user interface (`TUI`) will launch, and then you may interact using menu keys. 
A view of the `TUI` is presented below:
```
             
PLANS - PLANNING NATURE-BASED SOLUTIONS
Version: 3.0
This software is under the GNU GPL3.0 license
Source code repository: https://github.com/ipo-exe/plans3/





******************************* PLANS 3 *******************************


PLANS Menu
__________________
      Options Keys
 Open Project    1
  New Project    2
     Language    3
         Exit    e

>>> Chose key: 1

	>>> OK

Chosen:	Open Project

```

## Running `plans3` as a python package
Since `plans3` is a function-based software, you may desire to run very specific available functions or 
even embed it on your custom python code. 
This sets the user free to run `plans3` on any python IDE or cloud computing services.
### Functions documentation in `docstrings`
Most of relevant functions available in the modules has `dostrings`, which means that parameters and 
returns are fully described. To access a function `docstring` use the `help()` function:

`in:`
```python
from analyst import frequency  # import the frequency() function of the analyst module

help(frequency)  # call the help() function
```

`out:`
```
Help on function frequency in module analyst:

frequency(series)
    Frequency analysis on a given time series.
    :param series: 1-d numpy array
    :return: dictionary object with the following keys:
     'Pecentiles' - percentiles in % of array values (from 0 to 100 by steps of 1%)
     'Exeedance' - exeedance probability in % (reverse of percentiles: 100 - percentiles)
     'Frequency' - count of values on the histogram bin defined by the percentiles
     'Probability'- local bin empirical probability defined by frequency/count
     'Values' - values percentiles of bins
```

## Modules available on `plans3`

`plans3` is a function-based software. 
It contains a collection of modules of python functions, which are described below:

### `tools.py`
The specific backend tasks of `plans3`. 
It performs the silent routines of input, output and process execution.

### `analyst.py`
This module holds all data analysis basic functions.

### `backend.py`
General backend tasks. It performs the silent routines for the desktop application.  

### `evolution.py`
This module holds all evolutionary computing related basic functions.

### `geo.py`
This module holds all geoprocessing related basic functions.

### `hydrology.py`
This module holds all hydrolgy related models, incluing the model calibration routine.

### `input.py`
This module holds pre-processing input functions.

### `output.py`
This module holds post-processing output functions.

### `resample.py`
A collection of model functions and convenience functions for resampling time scale in time series analysis.

### `tui.py`
A terminal frontend of `plans3` is handled by the `tui.py` module. 
The interface is a simple terminal-based user interface that presents menus for the user.

### `visuals.py`
This module holds all built-in functions for creating data visualisations.


## Structure of a Plans2 Project

In the first time you run it, Plans2 automatically creates a directory in `C:/Plans2`. 
This is the standard root directory for all Plans2 projetcs. 
Projects are subdirectories inside the root folder, like `C:/Plans2/myproject/`. 
Once a project is named by the user, Plans2 create the project strucutre, which includes:
* A directory for datasets in `C:/Plans2/myproject/datasets/`
* A directory for storing execution files in `C:/Plans2/myproject/runbin`

Datasets are divided in `./datasets/observed` and `./datasets/projected`. 
`./datasets/observed` stores all data files for the "present" time of the water system being modelled. 

## The Terminal-based interface (TUI)

text

## IO files






