# dark-heatmap-generation

<div align="center">
    <img src="/assets/images/summary_figure.webp" height="320" alt="Summary figure">
</div>

**What the project does**:

This repository hosts MATLAB files designed to carry out ship detection on NovaSAR-1 imagery as well as data association with AIS data.

The outputs of these processes are used to generate "dark" heatmaps which identifies regions where frequent discrepancies between SAR and AIS datasets are observed.

<!-- _Note that this repository covers exclusively the X and does not cover the Y_. -->

**Why the project is useful**:

This project generates a higher-level product that shows the magnitude and scale of "dark" ships (non-AIS transmitting).

In other words, a multi-temporal analysis of satellite data is carried out to reveal areas where frequent mismatching between the SAR and AIS datasets occurs (called “dark hotspots”), and which represent areas of heightened IUU fishing risk.

This information can be used by authorities to make informed decisions and contribute towards effective Maritime Domain Awareness (MDA).

## Getting started

### Requirements

- [MATLAB R2023b](https://uk.mathworks.com/help/matlab/release-notes.html)
- [Mapping Toolbox](https://uk.mathworks.com/help/map/release-notes.html) <!-- To visualise the results, the following toolbox is recommended: -->
- [Statistics and Machine Learning Toolbox](https://uk.mathworks.com/help/stats/release-notes.html)
- [Computer Vision Toolbox](https://uk.mathworks.com/help/vision/release-notes.html)
- [Aerospace Toolbox](https://uk.mathworks.com/help/aerotbx/release-notes.html)
- [Text Analytics Toolbox](https://uk.mathworks.com/help/textanalytics/release-notes.html)
- [Sensor Fusion and Tracking Toolbox](https://uk.mathworks.com/help/fusion/release-notes.html)
- [Phased Array System Toolbox](https://uk.mathworks.com/help/phased/release-notes.html)

Download or clone this repository to your machine and open it in MATLAB.

### Datasets description

The script [s_detection_nv.m](s_detection_nv.m) described below can process all of NovaSAR-1's baseline acquisition modes, but be aware that the maritime mode may require significant computational resources.

The NovaSAR-1 data are Level-1 ScanSAR Ground Range Detected (SCD) products, meaning that the SAR data has been detected, multilooked and projected to ground range. For detailed information on NovaSAR-1's specifications, modes, and products, consult the [NovaSAR-1 User Guide](https://research.csiro.au/cceo/novasar/novasar-introduction/novasar-1-user-guide/).

The product folders include TIF files containing the amplitude values for each image band, as well as XML files that contain metadata for each image band.

Note that the NovaSAR products are not provided radiometrically calibrated.

<!-- Note that SNAP generates Sigma0 virtual bands after importing NovaSAR products when using the NovaSAR Product Reader plugin. -->

<!-- We have a MATLAB script available for NovaSAR radiometric calibration. -->

<!-- AIS data. Some level of pre-processing is done by the data provider. For example, Spire do... -->

### File description

<!-- NovaSAR-1 file structure. -->

Firstly, run the script [s_detection_nv.m](s_detection_nv.m). This script performs object detection on the SAR imagery and includes the following steps:

1. Pre-processing (metadata):
	- A geolocation grid is created from the image metadata and interpolated up to the full resolution of the image. This geolocation grid is later used to extract the centroids of the object detections.
2. Land masking:
	- To limit the detection to the sea, image data inside a land mask is excluded from the detection.
3. Radiometric calibration:
	- The SAR data is radiometrically calibrated using the parameters provided in the image metadata.
	- The intensity image band is created from the amplitude band and converted to the Decibels (dB) scale for help with visualising the results.
4. Detection:
	- A 2-D Constant False Alarm Rate (CFAR) detector using cell averaging is applied to the intensity image band.
	- The detector predominately uses a background window (or training band) size of 11 x 11 pixels, a guard band size of 8 x 8 pixels and the probability of false alarm is set to 10<sup>-7</sup>.
	- A binary image of the object detections is created. If needed, the binary image undergoes morphological opening, which involves erosion followed by dilation, to eliminate false positives. These false positives are often caused by sea clutter and the application of this process varies based on the specific image. <!-- , typically appearing as single-pixel anomalies, -->
	- The centroids and bounding boxes of the object detections are extracted. The length of the object detections is determined by extracting the maximum size of the object's bounding box.

Secondly, run the script [s_data_association_nv.m](s_data_association_nv.m) which carries out data processing and data association. This script includes the following steps:

1. Pre-processing (metadata):
	- The SAR footprint or bounding box is created from the image metadata.
	- The SAR image start time is also extracted from the metadata.
2. SAR data processing:
	1. Discrimination:
		- Object detections that are within 500 metres of offshore infrastructure are removed. The offshore infrastructure dataset being used is Paolo et al., [2024](https://globalfishingwatch.org/data-download/datasets/public-paper-industrial-activity-2024). As noted by the authors of this dataset, it is important to note that Sentinel-1 SAR data does not cover all of the open ocean.
		- Object detections that are located closer to one another than a specified distance threshold of 127.50 metres are merged. This is designed to resolve instances where the same vessel, typically a large one, is erroneously detected multiple times.
		- Object detections that are only one pixel in length are removed because distinguishing between true and false positives is too challenging at this size, and they can often be mistakenly detected due to sea clutter. This conservative approach helps to ensure accuracy. Similarly, object detections that are greater than 600 metres in length are removed.
	2. Ship classification: _TBD_.
3. AIS data processing:
	1. Spatio-temporal filtering:
		- The AIS data is temporally filtered relative to the date and time of the SAR image acquisition. Selecting a time interval is non-trivial as ships often do not comply with the technical standard. Therefore, the filter dynamically adjusts based on the reporting frequency of vessels within the AIS dataset.	This method involves analysing the reporting intervals of vessels and then setting the time window based on the variability of these intervals. <!-- in the vicinity of the SAR object detections -->
		- The AIS data is spatially filtered to a "guard" footprint to speed up subsequent processing, especially interpolation.
	2. Spatio-temporal alignment:
		- The AIS data is interpolated to the SAR image datetime. The implied speed and bearing are also calculated for subsequent processing steps.
		- An azimuth shift correction is applied to positions in the AIS data (as opposed to the SAR data) to account for the well-known effect observed in SAR imagery that, when an object is moving with a non-zero range velocity component, it will appear displaced in the azimuth direction.
	3. Spatial filtering:
		- The dataset is filtered again according to the spatial extent (or footprint) of the SAR image.
		- AIS data located within the SAR land mask (including the 250 metre buffer) are also removed.
		- Similar to Step 2. i., AIS data that are within 500 metres of offshore infrastructure are removed.
	4. AIS beacon identification:
		- AIS beacons are identified and excluded from the AIS data. This identification is achieved by analysing the vessel name and callsign fields in the AIS data, which often display distinctive patterns differentiating beacons from actual vessels.
	5. Data resolver:
		- The AIS dataset is cross-checked against a public vessel database to update missing data entries, including length, width and vessel type information.
4. Data association:
	- An _m_-best assignment technique is implemented to assign AIS data to SAR ship detections.
	- This technique ranks all the assignments in the order of increasing cost. The cost is defined as the distance for each AIS-SAR pair, which is computed using the geodesic distance.

<!-- ## Qualitative results -->

## Limitations

Known limitations include:

- The length of the object detections do not accurately represent the actual length of the vessel, as the detected pixels frequently do not encompass the vessel's full length. The length is used exclusively to remove object detections that are only one pixel in length. <!-- ![Example.](/assets/images/bbox.png) -->
- Removing object detections that are only one pixel in length can result in a minor increase in false negatives. However, this approach is adopted because the advantage of decreasing false positives is considered to outweigh the minor rise in false negatives.

These limitations are acknowledged and should be taken into consideration.

<!-- ## Next steps

Next steps include... (also see OneNote)

- For detection, it is probably more efficient to implement block processing on the SAR imagery.
- Rather than prioritising one polarisation, process all polarisation bands and merge the detection results.
- It would be worthwhile to validate the detection on an open dataset and retrieve performance metrics.
- The detection algorithm has not yet been validated on other satellite SAR data.
- Attempt to replicate SNAP's implementation of its fast geolocation grid interpolation method.
- Attempt to standardise the data fieldnames to facilitate data ingestion from different data providers.
- Improve discrimination: https://chat.openai.com/c/b66b9445-75f5-44d8-9471-b4a54fa3ccbc
- Improve the estimates of SAR-derived ship dimensions.

-->

<!-- ## References -->

## Further reading

- Rodger and Guida, [2020](https://www.mdpi.com/2072-4292/13/1/104).

<!-- ## How to Cite This Project -->

## License

The license is available in the [LICENSE file](LICENSE.txt) in this repository.
