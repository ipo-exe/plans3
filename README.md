#Plans3

Here we go!

# input files
## `calib_basin.asc`
**File type:** raster map

**Description:**
Boolean raster map of the area of calibration basin.

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: boolean (1.0 and 0.0).
*  Cells must be 1.0 where the the area is TRUE and 0.0 where the area is FALSE.



## `calib_catcha.asc`
**File type:** raster map

**Description:**
Raster map of catchment area (also known as flow accumulation) of the calibration basin.

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: squared meters.



## `calib_dem.asc`
**File type:** raster map

**Description:**
Raster map of Digital Elevation Model (DEM) of the calibration basin.

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.



## `calib_lulc.asc`
**File type:** raster map

**Description:**
Raster map of LULC (land use and land cover) for the calibration basin and calibration period. Each LULC class receives an index number defined in the lulc_calib_param file 

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: class index.



## `calib_lulc_param.txt`
**File type:** csv data frame

**Description:**
Data frame of LULC classes for the calibration basin.

**Requirements:**
*  Field separator: semicolon (` ;` ).
*  Decimal separator: period ( `.` ).
*  Mandatory fields:
*  `Id`: unique integer number of LULC class index.
*  `LULC`: one-word name of LULC class.
*  `f_Canopy`:  positive real number of factor of maximal effective canopy storage capacity in any units.
*  `f_RootDepth`: positive real number of fator of maximal effective root zone depth in any units.
*  `f_Depression`: positive real number of maximal effective surface depression storage capacity in any units.



## `calib_series.txt`
**File type:** csv time series

**Description:**
Daily time series of hydrologic data for the calibration basin in the calibration period.

**Requirements:**
*  Field separator: semicolon (` ;` ).
*  Decimal separator: period ( `.` ).
*  Date format: `YYYY-MM-DD`.
*  Mandatory fields:
*  `Date`: date of record.
*  `Prec`: daily accumulated precipitation in mm.
*  `Temp`: mean daily temperature in Celsius.
*  `Q`: mean daily specific flow in mm/day (Note: Q = 86400 [s/day] * Flow [m3/s] / BasinArea [m2]).
*  Optional fields:
*  `Flow`: mean daily flow in m3/s.



## `calib_soils.asc`
**File type:** raster map

**Description:**
Raster map of soils for the calibration basin. Each soil class receives an index number defined in the soils_calib_param file. 

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: class index.



## `calib_soils_param.txt`
**File type:** csv data frame

**Description:**
Data frame of soil classes for the calibration basin.

**Requirements:**
*  Field separator: semicolon (` ;` ).
*  Decimal separator: period ( `.` ).
*  Mandatory fields:
*  `Id`: unique integer number of soil class index.
*  `SoilClass`: one-word name of soil class.
*  `f_Ksat`:  positive real number of factor of maximal effective saturated hydraulic conductivity in any units.
*  `Porosity`: positive real number of soil porosity.



# derived files
## `calib_shru.asc`
**File type:** raster map

**Description:**
Raster map of Surface Hydrologic Response Units (SHRU) for the calibration basin. Each soil class receives an index number defined in the shru_calib_param file. 

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: class index.



## `calib_shru_param.txt`
**File type:** csv data frame

**Description:**
Data frame of Surface Hydrologic Response Units (SHRU) classes for the calibration basin.

**Requirements:**
*  Field separator: semicolon (` ;` ).
*  Decimal separator: period ( `.` ).
*  Mandatory fields:
*  `Id`: unique integer number of SHRU class index;
*  `SHRU`: one-word name of SHRU class.
*  `f_Ksat`:  positive real number of factor of maximal effective saturated hydraulic conductivity in any units.
*  `Porosity`: positive real number of soil porosity.
*  `f_Canopy`:  positive real number of factor of maximal effective canopy storage capacity in any units.
*  `f_RootDepth`: positive real number of fator of maximal effective root zone depth in any units.
*  `f_Depression`: positive real number of maximal effective surface depression storage capacity in any units.



## `calib_slope.asc`
**File type:** raster map

**Description:**
Raster map of terrain slope for the calibration basin.

**Requirements:**
*  Field separator: semicolon (` ;` ).
*  Decimal separator: period ( `.` ).
*  Mandatory fields:
*  `Id`: unique integer number of SHRU class index.
*  `SRHU`: name of SHRU class.
*  `f_Ksat`:  positive real number of factor of maximal effective saturated hydraulic conductivity in any units.
*  `Porosity`: positive real number of soil porosity.



## `calib_twi.asc`
**File type:** raster map

**Description:**
Raster map of the Topographic Wetness Index (TWI) for the calibration basin.

**Requirements:**
*  Must match the same size (rows and columns) of other related raster maps.
*  Grid cells must be squared.
*  Cells values units: TWI units.