# Quick guide to `plans3`

---
## Basic concepts of a `plans3` project

Text

### `aoi` - basin area of interest 

`aoi` is the basin area that you want to do the modelling.  

### `calib` - basin area of model calibration

`calib` is the gauged basin area where **flow data is available** for model calibration. 
If you are lucky, your `aoi` is the same of `calib`. But most of cases they may differ.
Warning: make sure your `calib` basin is **reasonably similar in terms of geology** of your `aoi` basin. 

![alt text](https://github.com/ipo-exe/plans3/blob/main/docs/figs/aoi_calib.PNG "aoi_calib")


### `lulc` - land use and land cover

`lulc` is the set of classes of land use and land cover. 
Classes may have attributes of management practices. For instance, you may want to separate _conventional croplands_
 and _conservation croplands_ in different `lulc` classes. One important feature of `lulc` is that
 **it can be changed by human action** (which means that you will need many maps of `lulc`!). 
 

![alt text](https://github.com/ipo-exe/plans3/blob/main/docs/figs/lulc.PNG "calib_lulc")

### `soils` - soils

`soils` is the set of classes of soils. These are considered static in human time scales. 
Note: `soils` classes not include the organic superficial horizon of soil, which actually is related to `lulc`.   

![Soils](https://github.com/ipo-exe/plans3/blob/main/docs/figs/soils.PNG "calib_soils")

### `shru` - surface hydrologic response units

`shru` stands for Surface Hydrologic Response Units. Those are patches in the landscape 
that behave similarly in terms of surface hydrology. By surface hydrology we refer to processes such as
canopy interceptation, surface pounding, infiltration and transpiration from the root zone.
   
You do not have to worry about these classes since `plans3` do the job of creating `shru` maps based
 on `soils` and `lulc`. But one thing is important: the number of possible `shru` is precisely the number of 
 `lulc` classes plus the number of `soil` classes - so 10 `lulc` classes and 10 `soils` classes would yield 100 `shru` 
 classes (10 x 10 = 100)!   
 
 ![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/shru.PNG "calib_shru")

### `etpat` - daily pattern of ET

`etpat` is the daily spatial pattern of actual evapotranspiration `ET` used for model pattern calibration.
The map units does not matter but must be positively correlated with `ET`. 
You may use remote-sense data sets, such as:
* Thermal band;
* Surface Temperature;
* ET (SEBAL);

![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/etpat.PNG "calib_shru")

### `dem` - digital elevation model

`dem` stands for Digital Elevation Model. It is a grid map of surface elevation.

![DEM](https://github.com/ipo-exe/plans3/blob/main/docs/figs/dem.PNG "calib_dem")


### `slope` - local terrain slope

`slope` is the estimated terrain angle in degrees. `plans3` can derive it from the `dem` map.

![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/slope.PNG "calib_slope")


### `catcha` - local catchment area

`catcha` is a map of local catchment area, also known as `flow accumulation`. 
It is expressed in squared meters. 
> Tip: You may derive this map from `dem` processing in a GIS Desktop 
Application, such as QGIS. If so, do not forget to conditionate the `dem` for hydrology analysis (i.e., remove sinks).

![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/catcha.PNG "calib_catcha")

### `fto` - local factor of soil water transmissivity

`fto` is a map of local factor of soil water transmissivity `to`. By factor we mean no units, it is just a value of how much 
the transmissivity of a patch of soil if proportional to the basin-wide effective soil transmissivity.

![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/fto.PNG "calib_fto")

### `twi` - topographic wetness index

`twi` stands for topographical wetness index. It just is a unit-less value of the propensity of soil get saturated by 
the water table. The higher the value, the more is the saturation propensity.

The local `twi` of Beven and Kirkby (1979) is defined by :
> twi = ln(catcha / (fto * tan(slope)))

But you may change that, as you please. Just make sure the highest values are related to more saturation 
so the local deficit `di` of water is mapped by the following equation:

> di = d_g + m * (twi_g - twi)

Where `d_g` is the basin-wide (global) water deficit, `twi_g` is average global `twi` and `m` is a scaling parameter. 


![SHRU](https://github.com/ipo-exe/plans3/blob/main/docs/figs/twi.PNG "calib_twi")



