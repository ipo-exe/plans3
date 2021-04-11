![alt text](https://github.com/ipo-exe/plans3/blob/main/docs/logo.png "Logo Title Text 1")
# `plans3` - **Planning Nature-based Solutions**

#### 🔴 Warning! ⚠ this project is _not_ a stable release!

> `plans3` is still under development. This `README.md` file is a pilot document.

#### What is `plans3`?
**Planning Nature-based Solutions** (`plans3`) is a modelling framework for planning the 
expansion of nature-based solutions in watersheds.

#### Why the "3" on `plans3`?
`plans` was born in 2018 within the scope of a master's degree research project. 
While `plans1` was just a handful of python scripts, `plans2` has an application-like structure. 
Now, `plans3` has deep changes in hydrology modelling.

## What is included in this repository

- [x] All files required to run `plans3`;
- [x] A directory called `./samples` with examples of input files;
- [x] A directory called `./docs` with documentation files;
- [x] In `./docs`: a markdown file called `iofiles.md` of I/O file documentation;
- [ ] In `./docs`: a markdown file called `guide.md` for a quick guide of `plans3` applications;
- [ ] A comprehensive `plans3_handbook.pdf` document.

## `python` and packages required

`plans3` is built on top of:
* [python 3.8](https://www.python.org/downloads/)
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [matplotlib](https://matplotlib.org/)
* [scipy](https://www.scipy.org/)

## Installing `plans3` on a local machine

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

## Running `plans3` as a desktop application

### The terminal-based user interface `TUI`
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

### The default `plans3` desktop project

In the first time you run the `TUI` application, `plans3` automatically creates a directory called `C:/Plans3`. 
This is the standard root directory for all desktop `plans3` _projects_. 
Projects are subdirectories inside the root folder, like `C:/Plans3/myproject/`. 

Once a project is named by the user, `plans3` create the project strucutre, which is divided in two main folders:
* A directory for datasets in `C:/Plans3/myproject/datasets/`
* A directory for storing output files in `C:/Plans3/myproject/runbin`

The `./datasets` directory is divided in `./datasets/observed` and `./datasets/projected`. 
* `./datasets/observed` stores all input and derived data files for the _observation period_ of the project. 
* `./datasets/projected` stores all input and derived data files for projected scenarios periods. 

The `./runbin` directory is divided in `./runbin/simulation` and `./runbin/optimization`. 
* `./runbin/simulation` stores all output data files of simulation procedures, which are located in time-stamped subfolders. 
* `./datasets/optimization` stores all output data files of optimization procedures, which are located in time-stamped subfolders. 


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

#### `analyst.py`
This module holds all data analysis basic functions.

#### `backend.py`
General backend tasks. It performs the silent routines for the desktop application.  

#### `evolution.py`
This module holds all evolutionary computing related basic functions.

#### `geo.py`
This module holds all geoprocessing related basic functions.

#### `hydrology.py`
This module holds all hydrolgy related models, incluing the model calibration routine.

#### `input.py`
This module holds pre-processing input functions.

#### `output.py`
This module holds post-processing output functions.

#### `resample.py`
A collection of model functions and convenience functions for resampling time scale in time series analysis.

#### `tools.py`
The specific backend tasks of `plans3`. 
It performs the silent routines of input, output and process execution.

#### `tui.py`
A terminal frontend of `plans3` is handled by the `tui.py` module. 
The interface is a simple terminal-based user interface that presents menus for the user.

#### `visuals.py`
This module holds all built-in functions for creating data visualisations.

## IO files

A full documentaion of IO files is provided by the `iofiles.md` document in `./docs`.

### Files formatting
Input and output files in `plans3` are all in open source format, namely `.txt` csv data frames
 and `.asc` raster maps. 
 
 🔴 Warning! ⚠ in csv files the field separator **must** be semicolon `;` and decimal plate separator **must** be period `.`.
 
 🔴 Warning! ⚠ the raster maps **must** be in a _projected_ coordinate reference system, which means 
  coordinates in _meters_ and _squared grid cells_. 
 
> tip: you may want to use [QGIS](https://qgis.org/en/site/) to **translate** your `.tif` 
>raster maps to `.asc` files and vice-versa. The only issue here is that `.asc` files do not store 
>the datum `EPSG` code so you will have to manually set it in the layer properties within QGIS (or another GIS application).
 
<a href="http://www.youtube.com/watch?feature=player_embedded&v=GDYraQg_otE
" target="_blank"><img src="http://img.youtube.com/vi/GDYraQg_otE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>