# Typhoon Track Plotting Script for GFS Data

## Description

This repository contains the `GFS/plot_track.py` Python script, designed to visualize typhoon track data, typically sourced from GFS (Global Forecast System) model outputs. The script reads specialized `.dat` files, plots the typhoon's best track, and includes any ensemble member tracks if they are present in the input data. The output is a PNG image showing the tracks on a map.

## Dependencies

The script requires the following Python libraries:

*   `numpy`: For numerical operations, especially handling arrays and NaN values.
*   `matplotlib`: For plotting.
*   `mpl_toolkits.basemap` (Matplotlib Basemap Toolkit): For creating maps with geographic projections.
*   `pandas`: Currently imported in the script but not actively used in the core data processing or plotting logic of the refactored version.
*   `argparse`: (Python Standard Library) For parsing command-line arguments.
*   `datetime`: (Python Standard Library) For handling date-time conversions, particularly for forecast hour calculations.

To install the necessary non-standard libraries, you can typically use pip:
```bash
pip install numpy matplotlib basemap pandas
```
Note: Installing Basemap can sometimes be complex due to its dependencies. Refer to the official Matplotlib Basemap documentation for detailed installation instructions if needed.

## Usage

To run the script, use the following command-line syntax:

```bash
python GFS/plot_track.py -i <input_file1.dat> [input_file2.dat ...]
```

**Arguments**:

*   `-i` or `--infile`: Specifies the input CWB (Central Weather Bureau) GFS Track filename(s). You can provide one or more `.dat` files. The script will process each file individually.

## Input File Format

The script expects input `.dat` files to follow a specific format:

1.  **Header Line 1**: Contains the initial Date-Time Group (DTG) and the number of typhoons in the file. The DTG is in `YYMMDDHH` format (Year, Month, Day, Hour).
    *   Example: `17090600    number of typhoons=  1`

2.  **Header Line 2**: Contains the typhoon number and its name.
    *   Example: `number =  1 typh-name= GUCHOL`

3.  **Data Lines**: Subsequent lines represent track data at different forecast hours (tau).
    *   Each line starts with the forecast hour (e.g., `000`, `006`, `012`).
    *   This is followed by pairs of latitude and longitude values.
    *   The first latitude/longitude pair is considered the **best track** point for that tau.
    *   Any subsequent latitude/longitude pairs on the same line are considered **ensemble member** track points for that tau.
    *   Missing data points (for latitude or longitude) are represented by `-99.999`.
    *   Example of a data line with best track and one ensemble member (with some missing data):
        `000  20.300 120.400  20.100 120.500 -99.999 -99.999`
        (Here, Tau=000, Best Track: Lat 20.3, Lon 120.4. Ensemble 1: Lat 20.1, Lon 120.5. Ensemble 2 would have had missing data if more pairs followed)

## Output

For each input `.dat` file, the script generates:

*   A PNG image file depicting the typhoon tracks on a map.
*   The output PNG file is saved in the **same directory** as the corresponding input `.dat` file.
*   The output filename is derived from the input filename by replacing the `.dat` extension with `.png`. For example, an input file named `trk17090600.dat` will produce `trk17090600.png`.

The plot will display the best track in blue and ensemble member tracks (if any) in red. The forecast period (initial DTG to final DTG) is also annotated on the map.
