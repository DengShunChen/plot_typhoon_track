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

---

## TWRF Typhoon Track Plotting Script (`TWRF/plot_twrf_track.py`)

## Description

The `TWRF/plot_twrf_track.py` script is designed to visualize typhoon track data from TWRF (Typhoon Weather Research and Forecasting) model output files. It can parse files containing multiple forecast blocks (potentially for different model runs or resolutions within the same file) and an optional 'best track' section. The script plots these tracks on a map.

## Dependencies

The script relies on the same Python libraries as `GFS/plot_track.py`:

*   `numpy`
*   `matplotlib`
*   `mpl_toolkits.basemap` (Matplotlib Basemap Toolkit)
*   `pandas` (Imported, though direct usage in plotting logic is minimal)
*   `argparse` (Python Standard Library)
*   `datetime` (Python Standard Library)

Refer to the GFS script's dependency section for installation notes, especially for Basemap.

## Usage

To run the TWRF plotting script, use the following command-line syntax:

```bash
python TWRF/plot_twrf_track.py -i <twrf_input_file1.dat> [twrf_input_file2.dat ...]
```

**Arguments**:

*   `-i` or `--infile`: Specifies the input TWRF DAT filename(s). You can provide one or more files. The script will process each file and generate a corresponding plot.

## Input File Format (TWRF)

The TWRF `.dat` files have a distinct structure:

*   **Multiple Forecast Blocks**: A single file can contain data for several forecast runs or model configurations.
    *   Each block begins with a header line indicating the forecast's starting date and time:
        `FORECAST DATE (YYMMDDHH) --<resolution>km`
        *   Example: `FORECAST DATE (16070406) --15km`
    *   This is followed by a line specifying the typhoon's name:
        `TYPHOON - <NAME>     PS`
        *   Example: `TYPHOON - NEPARTAK     PS`
    *   Subsequent lines within the block are data points:
        `ForecastHour  Lat  Lon  Pressure  [OptionalOtherColumns...]`
        *   Example: `  0    14.721   139.932   996.780`
        *   Missing numerical data is typically represented by `-999.000` or `-999.0`.

*   **Optional 'Best Track' Section**:
    *   May appear at the end of the file, initiated by a line containing only the word: `best`
    *   Data lines for the best track follow the format: `Lat  Lon  YYMMDDHH_DTG`
        *   Example: `       12.7     140.8       16070406`
    *   The best track section is terminated by a line containing: `99.9`

The `get_twrf_data` function in the script is designed to parse these structures, identifying each forecast block and the best track (if present) as separate tracks.

## Output

For each input TWRF `.dat` file, the script generates:

*   A PNG image file showing the plotted typhoon tracks (forecasts in red, best track in blue if available).
*   The output PNG file is saved in the **same directory** as the input `.dat` file.
*   The filename is generated by replacing the `.dat` extension of the input file with `.png` (e.g., `input.dat` becomes `input.png`). If the input filename does not contain `.dat`, `_TWRF_track.png` is appended.

The plot includes a title with the typhoon name and model ("TWRF"), and an annotation showing the overall period covered by the plotted data.
