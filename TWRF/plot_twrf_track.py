#!/usr/bin/env python
# coding: utf-8

# Purpose: Plot typhoon tracks from TWRF (Typhoon Weather Research and Forecasting) model data files.
# This script parses TWRF-specific DAT files, extracts multiple forecast tracks
# and an optional best track, then visualizes them on a map.
# Author: AI Assistant
# Date: 2024-07-27 (Updated with comments)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import argparse
from datetime import datetime, timedelta
# Note: The 'import numpy as np' line below 'timedelta' was redundant and removed.

# Function to safely convert to float or return np.nan
def safe_float(value, nan_val=-999.0):
    """
    Safely converts a value to a float. If conversion fails, returns np.nan.
    If the converted value matches nan_val, also returns np.nan.

    Args:
        value: The value to convert (usually a string).
        nan_val (float, optional): The specific float value that represents NaN.
                                   Defaults to -999.0.

    Returns:
        float: The converted float value, or np.nan if conversion fails or matches nan_val.
    """
    try:
        f_val = float(value)
        return np.nan if f_val == nan_val else f_val
    except ValueError:
        return np.nan

def get_twrf_data(filename):
    """
    Reads and parses typhoon track data from a TWRF model output file.

    The TWRF file can contain multiple forecast blocks and an optional best track section.
    - Forecast blocks start with "FORECAST DATE (YYMMDDHH) --<resolution>km"
      followed by "TYPHOON - <NAME>     PS". Data lines are "FH Lat Lon Pressure ...".
    - The best track section starts with "best" and data lines are "Lat Lon YYMMDDHH_DTG".
      It ends with "99.9".

    Args:
        filename (str): The path to the TWRF data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a single track
              (either a forecast block or the best track). Returns an empty list if
              the file cannot be read or contains no valid track data.
              Each dictionary has the following keys:
              - 'type' (str): Type of track, e.g., 'forecast_block_1', 'best_track'.
              - 'name' (str): Typhoon name associated with the track.
              - 'initial_dtg_str' (str or None): The initial DTG for forecast blocks,
                                                 None for the best track.
              - 'lats' (list of float): Latitudes, np.nan for missing.
              - 'lons' (list of float): Longitudes, np.nan for missing.
              - 'dtgs' (list of str): DTG strings for each point ("YYMMDDHH").
              - 'pressures' (list of float): Pressures, np.nan for missing.
                                            (Present for forecast blocks, empty for best_track).
    """
    all_tracks_data = []

    current_lats = []
    current_lons = []
    current_dtgs = []
    current_pressures = []
    current_typhoon_name = "UNKNOWN"
    current_initial_dtg_str = None

    parsing_state = None # Can be 'forecast' or 'best_track'
    forecast_block_counter = 0

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Check for new forecast block header
                if line.startswith("FORECAST DATE"):
                    # Finalize previous block if it exists
                    if parsing_state == 'forecast' and current_lats:
                        forecast_block_counter += 1
                        all_tracks_data.append({
                            'type': f'forecast_block_{forecast_block_counter}',
                            'name': current_typhoon_name,
                            'initial_dtg_str': current_initial_dtg_str,
                            'lats': list(current_lats),
                            'lons': list(current_lons),
                            'dtgs': list(current_dtgs),
                            'pressures': list(current_pressures)
                        })
                    # Reset for new block
                    current_lats, current_lons, current_dtgs, current_pressures = [], [], [], []
                    parsing_state = 'forecast'
                    try:
                        current_initial_dtg_str = line.split('(')[1].split(')')[0]
                    except IndexError:
                        print(f"Warning: Malformed FORECAST DATE line {line_num}: {line}. DTG might be incorrect.")
                        current_initial_dtg_str = "00000000" # Fallback DTG
                    # Next line is expected to be TYPHOON - NAME, continue to ensure it's processed by its own logic
                    continue

                # Check for TYPHOON name line (associated with a forecast block)
                # This should typically follow a "FORECAST DATE" line.
                if line.startswith("TYPHOON -") and parsing_state == 'forecast':
                    try:
                        current_typhoon_name = line.split('-')[1].split('PS')[0].strip()
                    except IndexError:
                        print(f"Warning: Malformed TYPHOON line {line_num}: {line}. Name might be incorrect.")
                        current_typhoon_name = "NAME_PARSE_ERROR"
                    continue

                # Check for start of 'best' track section
                if line == "best":
                    # Finalize previous forecast block if it exists
                    if parsing_state == 'forecast' and current_lats:
                        forecast_block_counter += 1
                        all_tracks_data.append({
                            'type': f'forecast_block_{forecast_block_counter}',
                            'name': current_typhoon_name, # Use last known typhoon name
                            'initial_dtg_str': current_initial_dtg_str,
                            'lats': list(current_lats),
                            'lons': list(current_lons),
                            'dtgs': list(current_dtgs),
                            'pressures': list(current_pressures)
                        })
                    # Reset for best track
                    current_lats, current_lons, current_dtgs, current_pressures = [], [], [], []
                    parsing_state = 'best_track'
                    # Note: The typhoon name for the best track will be the last `current_typhoon_name` encountered.
                    current_initial_dtg_str = None # Reset, not used for best track type in this way
                    continue

                # Check for end of 'best' track section
                if line == "99.9" and parsing_state == 'best_track':
                    if current_lats: # If there's best track data to save
                        all_tracks_data.append({
                            'type': 'best_track',
                            'name': current_typhoon_name, # Uses last typhoon name from forecast blocks
                            'initial_dtg_str': None,
                            'lats': list(current_lats),
                            'lons': list(current_lons),
                            'dtgs': list(current_dtgs), # DTGs are directly from file for best track
                            'pressures': [] # No pressure data for best track
                        })
                    parsing_state = None # End of best track processing
                    continue

                # --- Parse Data Lines ---
                parts = line.split()
                if not parts: # Should have been caught by line.strip() check, but as safeguard
                    continue

                if parsing_state == 'forecast':
                    if len(parts) >= 4 and current_initial_dtg_str: # FH, Lat, Lon, Pressure
                        try:
                            fh_str = parts[0]
                            # Check if forecast hour is a valid integer
                            if not fh_str.isdigit():
                                print(f"Warning: Non-integer forecast hour '{fh_str}' in line {line_num}: {line}. Skipping.")
                                continue
                            fh = int(fh_str)

                            lat = safe_float(parts[1])
                            lon = safe_float(parts[2])
                            pressure = safe_float(parts[3])

                            initial_dt = datetime.strptime(current_initial_dtg_str, "%y%m%d%H")
                            point_dt = initial_dt + timedelta(hours=fh)
                            point_dtg_str = point_dt.strftime("%y%m%d%H")

                            current_lats.append(lat)
                            current_lons.append(lon)
                            current_dtgs.append(point_dtg_str)
                            current_pressures.append(pressure)
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skipping malformed forecast data line {line_num}: {line} ({e})")
                    # elif parts : # Avoid printing warnings for lines that might be part of multi-line headers if any.
                         # print(f"Warning: Skipping unrecognized forecast data line {line_num}: {line}")

                elif parsing_state == 'best_track':
                    if len(parts) >= 3: # Lat, Lon, DTG_str
                        try:
                            lat = safe_float(parts[0])
                            lon = safe_float(parts[1])
                            dtg_str = parts[2] # Should be YYMMDDHH
                            # Validate DTG string format
                            datetime.strptime(dtg_str, "%y%m%d%H") # Will raise ValueError if not matching

                            current_lats.append(lat)
                            current_lons.append(lon)
                            current_dtgs.append(dtg_str)
                            # No pressure for best track
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Skipping malformed best track data line {line_num}: {line} ({e})")
                    # elif parts:
                        # print(f"Warning: Skipping unrecognized best_track data line {line_num}: {line}")

            # After loop, finalize any remaining forecast block data
            if parsing_state == 'forecast' and current_lats:
                forecast_block_counter += 1
                all_tracks_data.append({
                    'type': f'forecast_block_{forecast_block_counter}',
                    'name': current_typhoon_name,
                    'initial_dtg_str': current_initial_dtg_str,
                    'lats': list(current_lats),
                    'lons': list(current_lons),
                    'dtgs': list(current_dtgs),
                    'pressures': list(current_pressures)
                })
            # Finalize best track if it was the last thing being parsed and file ended without "99.9"
            # This is less ideal as "99.9" should be the proper terminator.
            elif parsing_state == 'best_track' and current_lats:
                 all_tracks_data.append({
                    'type': 'best_track',
                    'name': current_typhoon_name,
                    'initial_dtg_str': None,
                    'lats': list(current_lats),
                    'lons': list(current_lons),
                    'dtgs': list(current_dtgs),
                    'pressures': []
                })


    except FileNotFoundError:
        print(f"Error: TWRF data file {filename} not found.")
        return [] # Return empty list on file not found
    except Exception as e:
        print(f"An unexpected error occurred while reading {filename}: {e}")
        return [] # Return empty list on other errors

    if not all_tracks_data:
        print(f"Warning: No valid track data found in {filename}.")

    return all_tracks_data

def plot_map(typhoon_name, model_name, map_boundaries, default_boundaries=[80,0,190,50,120]):
    """
    Sets up and plots the basemap with coastlines, countries, and gridlines.

    Args:
        typhoon_name (str): Name of the typhoon for the plot title.
        model_name (str): Name of the model (e.g., "TWRF") for the plot title.
        map_boundaries (list): List of map boundary coordinates:
                               [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, lon_0_central].
                               Used to define the map projection area.
        default_boundaries (list, optional): Default map boundaries if not provided.
                                             Defaults to [80,0,190,50,120].
    """
    fig, ax = plt.subplots(figsize=(16,8))

    # Setup Lambert Conformal Conic map projection.
    m = Basemap(llcrnrlon=map_boundaries[0], llcrnrlat=map_boundaries[1],
                urcrnrlon=map_boundaries[2], urcrnrlat=map_boundaries[3],
                lon_0=map_boundaries[4],
                projection='lcc', lat_1=10., lat_2=40.,
                resolution ='l', area_thresh=1000., ax=ax)

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='white') # Fill ocean areas
    m.fillcontinents(color='burlywood',lake_color='white') # Fill continents and lakes

    # Draw parallels (latitude lines) and meridians (longitude lines)
    m.drawparallels(np.arange(15,70,20),labels=[1,1,0,0]) # Labels: [left,right,top,bottom]
    m.drawmeridians(np.arange(80,190,20),labels=[0,0,0,1])

    # Set plot title - Adjusted for TWRF context
    twrftitle = f'TWRF ({model_name}) Typhoon Tracks - {typhoon_name}'
    plt.title(twrftitle)

    return fig, ax, m

def set_map(all_lats_np, all_lons_np, all_dtgs_np, dtg_end_val):
  """
  Calculates optimal map boundaries based on the provided track data.

  Args:
      all_lats_np (np.array): NumPy array of all latitudes (best track + ensembles).
      all_lons_np (np.array): NumPy array of all longitudes.
      all_dtgs_np (np.array): NumPy array of corresponding DTG strings.
      dtg_end_val (str): The end DTG string to filter data points up to this time.

  Returns:
      list: Calculated map boundaries [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, lon_0_central].
            Returns default boundaries if no valid data points are found.
  """
  # Filter out NaN values from latitude and longitude arrays.
  # DTGs corresponding to NaN lat/lon are also removed by this indexing.
  valid_points_idx = ~np.isnan(all_lats_np) & ~np.isnan(all_lons_np)

  filtered_lats = all_lats_np[valid_points_idx]
  filtered_lons = all_lons_np[valid_points_idx]
  filtered_dtgs = all_dtgs_np[valid_points_idx]

  # Apply time boundary condition: select points up to or at dtg_end_val.
  # DTG strings (e.g., "17090600") can be compared lexicographically for this format.
  time_bound_idx = filtered_dtgs <= dtg_end_val

  final_lats = filtered_lats[time_bound_idx]
  final_lons = filtered_lons[time_bound_idx]

  # If no valid data points remain after filtering, return default boundaries.
  if final_lats.size == 0 or final_lons.size == 0:
    print('Warning: No valid data points found for map boundary calculation. Using default boundaries.')
    return [80., 0., 190., 50., 120.] # Default values also used in plot_map's signature

  # Determine min/max latitude and longitude from the filtered data.
  lat_max = final_lats.max()
  lat_min = final_lats.min()
  lon_max = final_lons.max()
  lon_min = final_lons.min()

  print('Calculated map boundary raw lat max/min = ',lat_max,lat_min)
  print('Calculated map boundary raw lon max/min = ',lon_max,lon_min)

  # Define map view with some padding around the track extents.
  expendx = 5. # Longitude padding
  expendy = 3. # Latitude padding
  llcrnrlon = float(int(lon_min - expendx)) # Lower-left corner longitude
  llcrnrlat = float(int(lat_min - expendy)) # Lower-left corner latitude
  urcrnrlon = float(int(lon_max + expendx)) # Upper-right corner longitude
  urcrnrlat = float(int(lat_max + expendy)) # Upper-right corner latitude
  lon_0 = lon_min + 0.5*(lon_max-lon_min)   # Central longitude for projection

  return [llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,lon_0]

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Plot TWRF Typhoon Track Data from DAT files.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--infile', help='Input TWRF DAT filename(s)',
                        type=str, nargs='+', required=True)
    args = parser.parse_args()
    filenames = args.infile

    print(f"Starting TWRF track plotting for: {filenames}")

    # --- File Loop ---
    for filename in filenames:
        print(f"Processing file: {filename}")

        # --- Data Loading ---
        # get_twrf_data returns a list of track dictionaries.
        tracks_data = get_twrf_data(filename)

        # Skip file if no data was loaded (e.g., file not found, parse error, or empty file)
        if not tracks_data:
            print(f"Warning: No track data loaded from {filename}. Skipping.")
            continue

        # --- Data Preparation for Map Boundaries (set_map) ---
        # Aggregate all points from all tracks in the file to determine overall map boundaries.
        all_lats_list = []
        all_lons_list = []
        all_dtgs_list = []

        # Use the name from the first track for the overall plot title (assuming consistency within a file).
        typhoon_name_from_data = "UnknownTyphoon"
        if tracks_data: # Should be true if we haven't continued
            typhoon_name_from_data = tracks_data[0].get('name', "UnknownTyphoon")

        for track in tracks_data:
            all_lats_list.extend(track.get('lats', []))
            all_lons_list.extend(track.get('lons', []))
            all_dtgs_list.extend(track.get('dtgs', []))

        # If, after checking all tracks, there are no latitude points, skip.
        if not all_lats_list:
            print(f"Warning: No latitude data points found in {filename} after processing all tracks. Skipping.")
            continue

        # Convert aggregated lists to NumPy arrays for filtering and calculations.
        all_lats_np = np.asarray(all_lats_list, dtype=np.float64)
        all_lons_np = np.asarray(all_lons_list, dtype=np.float64)
        all_dtgs_np = np.asarray(all_dtgs_list) # DTGs are strings

        # Filter out NaNs from the aggregated data before passing to set_map.
        nan_mask = ~np.isnan(all_lats_np) & ~np.isnan(all_lons_np)
        valid_lats_for_map = all_lats_np[nan_mask]
        valid_lons_for_map = all_lons_np[nan_mask]
        valid_dtgs_for_map = all_dtgs_np[nan_mask]

        # If no valid (non-NaN) points remain, skip plotting for this file.
        if valid_lats_for_map.size == 0:
            print(f"Warning: No valid (non-NaN) data points for map boundary setting in {filename}. Skipping.")
            continue

        # Determine the overall min and max DTG for annotations and map time bounds.
        min_dtg_str_overall = min(valid_dtgs_for_map) if valid_dtgs_for_map.size > 0 else "N/A"
        max_dtg_str_overall = max(valid_dtgs_for_map) if valid_dtgs_for_map.size > 0 else "N/A"

        # --- Map Setup ---
        # Calculate map boundaries based on all valid points up to the latest DTG.
        map_boundaries = set_map(valid_lats_for_map, valid_lons_for_map, valid_dtgs_for_map, max_dtg_str_overall)

        model_name = "TWRF" # Model name is fixed for this script context.

        # Create the map figure and axes.
        fig, ax, m = plot_map(typhoon_name_from_data, model_name, map_boundaries)
        if fig is None: # Check if map creation failed (plot_map current version doesn't do this, but good practice)
            print(f"Error: Failed to create map for {filename}. Skipping.")
            continue

        # --- Plotting Tracks ---
        # Iterate through each track (forecast block or best track) again to plot them individually.
        best_track_plotted_for_legend = False # To ensure "Best Track" label appears only once in legend.
        for track_idx, track in enumerate(tracks_data):
            track_lats_np = np.asarray(track.get('lats', []), dtype=np.float64)
            track_lons_np = np.asarray(track.get('lons', []), dtype=np.float64)
            track_dtgs_np = np.asarray(track.get('dtgs', []))

            if track_lats_np.size == 0: # Skip if this specific track is empty
                continue

            # Filter NaNs for the current track.
            current_nan_mask = ~np.isnan(track_lats_np) & ~np.isnan(track_lons_np)
            plot_lats = track_lats_np[current_nan_mask]
            plot_lons = track_lons_np[current_nan_mask]
            plot_dtgs = track_dtgs_np[current_nan_mask]

            if plot_lats.size == 0: # Skip if no valid points after NaN filter
                continue

            # Filter points by the overall latest DTG for consistent plotting up to max_dtg_str_overall.
            time_mask_plot = plot_dtgs <= max_dtg_str_overall
            final_plot_lats = plot_lats[time_mask_plot]
            final_plot_lons = plot_lons[time_mask_plot]

            if final_plot_lats.size == 0: # Skip if no points after time filter
                continue

            # Determine plot properties based on track type.
            color = 'red'
            label = None
            linewidth = 0.8
            alpha = 0.7

            if track.get('type') == 'best_track':
                color = 'blue'
                linewidth = 1.5
                alpha = 1.0
                if not best_track_plotted_for_legend:
                    label = 'Best Track'
                    best_track_plotted_for_legend = True
            else: # Forecast blocks
                # Optional: Could label each forecast block, e.g., label=track.get('type')
                pass

            # Convert lat/lon to map projection coordinates and plot.
            x, y = m(final_plot_lons, final_plot_lats)
            m.plot(x, y, color=color, label=label, linewidth=linewidth, alpha=alpha)

        # --- Final Touches for the Plot ---
        # The main title is set in plot_map. Additional title elements can be added here if needed.
        # Example: ax.set_title(f"{model_name} Tracks for {typhoon_name_from_data} ({min_dtg_str_overall} - {max_dtg_str_overall})", loc='center')

        # Add a period annotation (e.g., "YYMMDDHH - YYMMDDHH") to the plot.
        if min_dtg_str_overall != "N/A" and max_dtg_str_overall != "N/A":
            cperiod = f"{min_dtg_str_overall} - {max_dtg_str_overall}"
            # Position annotation slightly offset from the lower-left corner of the map boundaries.
            x2, y2 = m(map_boundaries[0] + 1, map_boundaries[1] + 1.)
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9) # Annotation box style.
            ax.annotate(cperiod, xy=(x2, y2), color='navy', ha='left', va='bottom', bbox=bbox_props)

        # Add legend if any track was plotted with a label (typically the 'Best Track').
        if best_track_plotted_for_legend:
            legend = ax.legend(loc='best', shadow=True) # 'best' location automatically finds good spot.
            if legend: # Check if legend object was created (i.e., if there were labeled artists)
                legend.get_frame().set_facecolor('w') # Set legend background to white.

        # Determine output filename and save the plot.
        output_filename = filename.replace('.dat', '.png')
        if '.dat' not in filename.lower(): # Fallback if input filename doesn't end with .dat
             output_filename = f"{filename}_TWRF_track.png"

        try:
            plt.savefig(output_filename, dpi=150) # Save figure with specified DPI.
            print(f"Plot saved to {output_filename}")
        except Exception as e:
            print(f"Error saving plot to {output_filename}: {e}")

        plt.close(fig) # Close the figure to free up memory.

    print("TWRF track plotting finished.")
