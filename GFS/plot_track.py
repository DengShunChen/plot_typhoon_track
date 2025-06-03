#!/usr/bin/env python
#--------------------------------------------------
# Purpose: Plot GFS typhoon tracks from CWB GFS track data files.
# This script reads typhoon track data, including best track and ensemble members,
# and generates a map plot showing these tracks.
#
# Author: Deng-Shun Chen
# All rights reserved
# 2018-05-24
#
# Program history:
#   2018-05-25  Deng-Shun  Created
#   (Recent refactoring by an AI assistant in 2024)
#
#--------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import pandas as pd # pandas is imported but not explicitly used in the current version of these functions.
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from datetime import datetime,timedelta
# numpy is imported again here, but already imported above. This is redundant.
# import numpy as np # Ensure numpy is imported -> This will be covered by the top-level import.

def get_data(filename):
    """
    Reads typhoon track data from a CWB GFS track file.

    Expected file format:
    Line 1: <YYMMDDHH>    number of typhoons=  <N>  (Initial DTG, number of typhoons)
    Line 2: number =  <ID> typh-name= <NAME>       (Typhoon number, typhoon name)
    Subsequent lines: <TAU>  <LAT_BT> <LON_BT>  <LAT_E1> <LON_E1> ...
                      (Forecast hour, Best Track Lat/Lon, Ensemble Lat/Lon pairs)
                      Missing values are denoted by -99.999.

    Args:
        filename (str): The path to the CWB GFS track data file.

    Returns:
        list: A list containing parsed track data:
            0: ensemble_tracks_lats (list of lists of floats): Latitudes for each ensemble member.
            1: ensemble_tracks_lons (list of lists of floats): Longitudes for each ensemble member.
            2: ensemble_tracks_dtgs (list of lists of strings): DTG strings for each point of each ensemble member.
            3: best_track_lats (list of floats): Latitudes for the best track.
            4: best_track_lons (list of floats): Longitudes for the best track.
            5: best_track_dtgs (list of strings): DTG strings for each point of the best track.
            6: dtg_ini_str (str): Initial DTG string from the file header (e.g., "17090600").
            7: dtg_end_str (str): Calculated end DTG string (from initial DTG + last tau).
            8: typhoon_name_str (str): Typhoon name from the file header (e.g., "GUCHOL").
            9: model_name_str (str): Model name (hardcoded as "GFS_UNKNOWN" as it's not in the file).
    """
    typhoon_name_str = ""
    model_name_str = "GFS_UNKNOWN"  # Placeholder, as model name is not in the data file
    dtg_ini_str = ""
    dtg_end_str = ""
    
    best_track_lats = []
    best_track_lons = []
    best_track_dtgs = []
    
    ensemble_tracks_lats = []
    ensemble_tracks_lons = []
    ensemble_tracks_dtgs = []
    
    taus = [] # List to store forecast hours (tau) from the file

    try:
        f = open(filename, 'r')
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        # Return empty/default values as per specified structure
        return [[], [], [], [], [], [], "", "", "", model_name_str]

    # Parse Line 1: Initial DTG, number of typhoons
    header_line1 = f.readline().strip().split()
    if not header_line1:
        f.close()
        print(f"Error: File {filename} is empty or header line 1 is missing.")
        return [[], [], [], [], [], [], "", "", "", model_name_str]
        
    dtg_ini_str = header_line1[0]
    # num_typhoons = int(header_line1[4]) # This info is parsed but not directly used in the returned structure.

    # Parse Line 2: Typhoon number, typhoon name
    header_line2 = f.readline().strip().split()
    if not header_line2 or len(header_line2) < 5 : # Basic check for enough parts
        f.close()
        print(f"Error: File {filename} header line 2 is malformed: {header_line2}")
        return [[], [], [], [], [], [], dtg_ini_str, "", typhoon_name_str, model_name_str]

    # Extract typhoon name. Example: "typh-name=GUCHOL"
    # Extract typhoon name. Example: "typh-name=GUCHOL"
    # header_line2 is already split, e.g. ['number', '=', '1', 'typh-name=', 'GUCHOL']
    name_part_index = -1
    for i, part in enumerate(header_line2):
        if "typh-name=" in part: # part could be "typh-name=GUCHOL" or "typh-name="
            name_part_index = i
            break
    
    if name_part_index != -1:
        current_part = header_line2[name_part_index] # e.g., "typh-name=" or "typh-name=GUCHOL"
        split_current_part = current_part.split('=', 1) # Split only on the first '=' (safer)
        
        # Case A: Name is attached to the token, e.g., "typh-name=GUCHOL"
        if len(split_current_part) > 1 and split_current_part[1] != '':
            typhoon_name_str = split_current_part[1]
        # Case B: Token is "typh-name=", and name is the next element in header_line2 list
        elif (name_part_index + 1) < len(header_line2):
            typhoon_name_str = header_line2[name_part_index + 1]
        else:
            # If "typh-name=" token is found but the structure doesn't match A or B
            print(f"Warning: Found 'typh-name=' token at index {name_part_index} but structure is unexpected: {header_line2}")
            typhoon_name_str = "UNKNOWN_TYPHOON_FORMAT_ERROR"
    else:
        # Fallback if "typh-name=" token itself is not found in any part of header_line2.
        # This part attempts to guess based on a fixed position.
        print(f"Warning: 'typh-name=' token not found by direct search in header line 2: {header_line2}.")
        # Check if index 4 looks like "name=value" or just "value"
        if len(header_line2) > 4: # If there is an element at index 4
            possible_name_part = header_line2[4]
            if '=' in possible_name_part and len(possible_name_part.split('=')) > 1:
                 typhoon_name_str = possible_name_part.split('=')[1] # Assumes "something=NAME"
            else:
                 typhoon_name_str = possible_name_part # Assumes "NAME"
        else:
            print(f"Could not determine typhoon name from header line 2 using fallback logic either.")
            typhoon_name_str = "UNKNOWN_TYPHOON_NO_TOKEN"

    # Strip potential extra quotes if name was extracted like '"GUCHOL"' or 'GUCHOL '
    typhoon_name_str = typhoon_name_str.strip('\'" ')

    first_data_line = True # Flag to initialize ensemble lists based on the first data line
    last_tau_hours = 0     # To determine dtg_end_str

    # Process data lines
    for line_content in f:
        parts = line_content.strip().split()
        if not parts: # Skip empty lines
            continue

        tau_str = parts[0]
        try:
            current_tau_hours = int(tau_str)
        except ValueError:
            print(f"Warning: Could not parse tau value '{tau_str}' as integer. Skipping line: {line_content.strip()}")
            continue
            
        taus.append(current_tau_hours)
        last_tau_hours = current_tau_hours

        # Calculate DTG for this set of points (e.g., "17090600" + 6 hours -> "17090606")
        # DTG format is YYMMDDHH. datetime correctly handles "YY" for 20YY.
        try:
            start_dt = datetime.strptime(dtg_ini_str, "%y%m%d%H")
            current_dt = start_dt + timedelta(hours=current_tau_hours)
            dtg_for_points = current_dt.strftime("%y%m%d%H")
        except ValueError:
            # Fallback if dtg_ini_str is not in the expected format
            print(f"Warning: Could not parse dtg_ini_str: '{dtg_ini_str}'. Using fallback DTG format for points.")
            dtg_for_points = f"{dtg_ini_str}_+{tau_str}H"


        # Best track data (first lat/lon pair after tau)
        if len(parts) >= 3: # Need at least tau, lat_bt, lon_bt
            lat_bt = float(parts[1]) if parts[1] != "-99.999" else np.nan
            lon_bt = float(parts[2]) if parts[2] != "-99.999" else np.nan
            best_track_lats.append(lat_bt)
            best_track_lons.append(lon_bt)
            best_track_dtgs.append(dtg_for_points)
        else:
            # If not enough parts for best track, append NaN and DTG
            best_track_lats.append(np.nan)
            best_track_lons.append(np.nan)
            best_track_dtgs.append(dtg_for_points)
            # Ensemble processing below will be skipped or add NaNs if initialized


        # Ensemble data (subsequent lat/lon pairs)
        ensemble_pairs = parts[3:] # Remaining parts are ensemble lat/lon pairs
        num_ensemble_members_this_line = len(ensemble_pairs) // 2

        if first_data_line and num_ensemble_members_this_line > 0:
            # Initialize lists for each ensemble member based on the count in the first data line
            ensemble_tracks_lats = [[] for _ in range(num_ensemble_members_this_line)]
            ensemble_tracks_lons = [[] for _ in range(num_ensemble_members_this_line)]
            ensemble_tracks_dtgs = [[] for _ in range(num_ensemble_members_this_line)]
            first_data_line = False
        # Handle case where first few lines might have no ensembles, then ensembles appear
        elif not ensemble_tracks_lats and num_ensemble_members_this_line > 0:
            ensemble_tracks_lats = [[] for _ in range(num_ensemble_members_this_line)]
            ensemble_tracks_lons = [[] for _ in range(num_ensemble_members_this_line)]
            ensemble_tracks_dtgs = [[] for _ in range(num_ensemble_members_this_line)]
            # first_data_line might remain true if it was true before, effectively meaning this is the first line
            # that *actually* defines the number of ensembles.

        current_line_member_idx = 0
        for i in range(0, len(ensemble_pairs), 2): # Process pairs of lat/lon
            if current_line_member_idx < len(ensemble_tracks_lats): # Check against initialized member count
                lat_ens = float(ensemble_pairs[i]) if ensemble_pairs[i] != "-99.999" else np.nan
                # Check for i+1 to prevent IndexError if an odd number of ensemble values exist
                lon_ens = float(ensemble_pairs[i+1]) if (i+1 < len(ensemble_pairs) and ensemble_pairs[i+1] != "-99.999") else np.nan
                
                ensemble_tracks_lats[current_line_member_idx].append(lat_ens)
                ensemble_tracks_lons[current_line_member_idx].append(lon_ens)
                ensemble_tracks_dtgs[current_line_member_idx].append(dtg_for_points)
            current_line_member_idx += 1
        
        # If this line has fewer members than established, pad with NaNs for this tau step
        if len(ensemble_tracks_lats) > 0 : # Only if ensembles have been initialized
            for member_idx in range(current_line_member_idx, len(ensemble_tracks_lats)):
                ensemble_tracks_lats[member_idx].append(np.nan)
                ensemble_tracks_lons[member_idx].append(np.nan)
                ensemble_tracks_dtgs[member_idx].append(dtg_for_points)

    f.close()

    # Calculate dtg_end_str from the initial DTG and the last forecast hour (tau)
    if taus: # If any data lines were processed
        try:
            start_dt = datetime.strptime(dtg_ini_str, "%y%m%d%H")
            end_dt = start_dt + timedelta(hours=last_tau_hours)
            dtg_end_str = end_dt.strftime("%y%m%d%H")
        except ValueError:
             # Fallback if dtg_ini_str was bad
             dtg_end_str = f"{dtg_ini_str}_+{last_tau_hours}H" if dtg_ini_str else f"UNKNOWN_DTG_END_T{last_tau_hours}"
    else: # No data lines, dtg_end_str might be same as dtg_ini_str or empty if dtg_ini_str was also problematic
        dtg_end_str = dtg_ini_str if dtg_ini_str else ""

    # Return all parsed data in a specific list structure
    return [
        ensemble_tracks_lats, ensemble_tracks_lons, ensemble_tracks_dtgs,
        best_track_lats, best_track_lons, best_track_dtgs,
        dtg_ini_str, dtg_end_str, typhoon_name_str, model_name_str
    ]

def plot_map(typhoon_name, model_name, map_boundaries, default_boundaries=[80,0,190,50,120]):
    """
    Sets up and plots the basemap with coastlines, countries, and gridlines.

    Args:
        typhoon_name (str): Name of the typhoon for the plot title.
        model_name (str): Name of the model (e.g., "GFS") for the plot title.
        map_boundaries (list): List of map boundary coordinates:
                               [llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, lon_0_central].
                               Used to define the map projection area.
        default_boundaries (list, optional): Default map boundaries if not provided.
                                             Defaults to [80,0,190,50,120].
    """
    # If map_boundaries is not provided or is None, use default_boundaries.
    # This check is technically not needed if map_boundaries always has a value from set_map,
    # but good for robustness if plot_map could be called from elsewhere.
    # current set_map returns defaults if it cannot compute, so map_boundaries should always be valid.
    # For simplicity, the parameter map_boundaries in the definition takes default_boundaries if not passed,
    # so we just use map_boundaries directly.
    
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
  
    # Set plot title
    twrftitle = f'CWB GFS({model_name}) Typhoon Tracks - {typhoon_name}'
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
  parser = ArgumentParser(description = 'Plot CWB GFS typhoon track data from specified file(s).',
                          formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i','--infile',help='Input CWB GFS Track filename(s)',
                      type=str,nargs='+',required=True)
  args = parser.parse_args()
  filenames = args.infile
  
  # --- Process Each Input File ---
  for filename in filenames:
    print(f"Processing file: {filename}")
    
    # --- Get Track Data ---
    # Data is returned as a list, see get_data docstring for structure.
    data = get_data(filename)
    
    # Unpack data from the list returned by get_data
    ensemble_lats_raw = data[0] # List of lists of latitudes for ensemble tracks
    ensemble_lons_raw = data[1] # List of lists of longitudes for ensemble tracks
    ensemble_dtgs_raw = data[2] # List of lists of DTG strings for ensemble tracks
    best_lats_raw = data[3]     # List of latitudes for the best track
    best_lons_raw = data[4]     # List of longitudes for the best track
    best_dtgs_raw = data[5]     # List of DTG strings for the best track
    dtg_ini_str = data[6]       # Initial DTG string (e.g., "17090600")
    dtg_end_str = data[7]       # End DTG string (e.g., "17091100")
    typhoon_name = data[8]      # Typhoon name (e.g., "GUCHOL")
    model_name = data[9]        # Model name (e.g., "GFS_UNKNOWN")

    # Skip file if essential data is missing (e.g., no typhoon name or no best track data)
    if not typhoon_name or not best_lats_raw:
        print(f"Skipping {filename} due to missing typhoon name or essential best track data.")
        continue
    
    # --- Prepare Data for Map Boundary Calculation (set_map) ---
    # Convert best track data to NumPy arrays for easier manipulation
    best_lats_np = np.asarray(best_lats_raw, dtype=np.float64)
    best_lons_np = np.asarray(best_lons_raw, dtype=np.float64)
    best_dtgs_np = np.asarray(best_dtgs_raw)

    # Combine best track and all ensemble tracks into flat lists for boundary calculation
    all_lats_list = list(best_lats_np.flatten())
    all_lons_list = list(best_lons_np.flatten())
    all_dtgs_list = list(best_dtgs_np.flatten())

    for track_lats in ensemble_lats_raw: all_lats_list.extend(track_lats)
    for track_lons in ensemble_lons_raw: all_lons_list.extend(track_lons)
    for track_dtgs in ensemble_dtgs_raw: all_dtgs_list.extend(track_dtgs)
    
    # Convert combined lists to NumPy arrays
    temp_all_lats_np = np.asarray(all_lats_list, dtype=np.float64)
    temp_all_lons_np = np.asarray(all_lons_list, dtype=np.float64)
    temp_all_dtgs_np = np.asarray(all_dtgs_list) # DTGs are strings

    # Filter out NaNs from combined data (based on lats/lons) before passing to set_map
    nan_mask_formap = np.isnan(temp_all_lats_np) | np.isnan(temp_all_lons_np)
    all_lats_for_set_map = temp_all_lats_np[~nan_mask_formap]
    all_lons_for_set_map = temp_all_lons_np[~nan_mask_formap]
    all_dtgs_for_set_map = temp_all_dtgs_np[~nan_mask_formap]
    
    # --- Determine Map Boundaries ---
    map_boundaries = set_map(all_lats_for_set_map, all_lons_for_set_map, all_dtgs_for_set_map, dtg_end_str)
    
    # Print information for the current plot
    print('Initial/End DTG = ', dtg_ini_str, dtg_end_str)
    print('Typhoon Name = ', typhoon_name)
    print('Model Name = ', model_name)
    print('Map View Boundaries (lllon,lllat,urlon,urlat,lon0) = ', map_boundaries)
    
    # --- Plot Map and Tracks ---
    fig, ax, m = plot_map(typhoon_name, model_name, map_boundaries)
    
    # Plot ensemble forecast tracks
    lcolor_ens = 'r' # Color for ensemble tracks
    for i in range(len(ensemble_lats_raw)): # Iterate through each ensemble member
      # Convert current ensemble member's data to NumPy arrays
      current_ens_lats = np.asarray(ensemble_lats_raw[i], dtype=np.float64)
      current_ens_lons = np.asarray(ensemble_lons_raw[i], dtype=np.float64)
      current_ens_dtgs = np.asarray(ensemble_dtgs_raw[i])

      # Filter out NaNs for this specific ensemble member track
      nan_mask_ens = ~np.isnan(current_ens_lats) & ~np.isnan(current_ens_lons)
      plot_lats_ens = current_ens_lats[nan_mask_ens]
      plot_lons_ens = current_ens_lons[nan_mask_ens]
      plot_dtgs_ens = current_ens_dtgs[nan_mask_ens]
      
      # Filter by dtg_end_str (plot points up to or at the end DTG)
      time_mask_ens = plot_dtgs_ens <= dtg_end_str
      final_lats_ens = plot_lats_ens[time_mask_ens]
      final_lons_ens = plot_lons_ens[time_mask_ens]
      
      # Plot if there are any valid points remaining
      if final_lats_ens.size > 0:
          x_ens, y_ens = m(final_lons_ens, final_lats_ens) # Convert to map projection coordinates
          m.plot(x_ens, y_ens, color=lcolor_ens, alpha=0.7) # Plot with some transparency
    
    # Plot best track
    lcolor_best = 'b' # Color for best track
    # Best track data (best_lats_np, etc.) already converted to NumPy arrays earlier
    nan_mask_best = ~np.isnan(best_lats_np) & ~np.isnan(best_lons_np)
    plot_lats_best = best_lats_np[nan_mask_best]
    plot_lons_best = best_lons_np[nan_mask_best]
    plot_dtgs_best = best_dtgs_np[nan_mask_best] # DTGs are assumed to be valid if lat/lon are

    # Filter by dtg_end_str
    time_mask_best = plot_dtgs_best <= dtg_end_str
    final_lats_best = plot_lats_best[time_mask_best]
    final_lons_best = plot_lons_best[time_mask_best]

    # Plot if there are any valid points remaining
    if final_lats_best.size > 0:
        x_best, y_best = m(final_lons_best, final_lats_best) # Convert to map projection coordinates
        m.plot(x_best, y_best, color=lcolor_best, linewidth=1.5, label='Best Track')
    
    # --- Add Period Annotation and Legend ---
    cperiod = str(dtg_ini_str)+' - '+str(dtg_end_str) # Create period string
    # Position for annotation (slightly offset from map corner)
    x2,y2 = m(map_boundaries[0]+1, map_boundaries[1]+1.)
    bbox_props = dict(boxstyle="round",fc="w", ec="0.5", alpha=0.9) # Bounding box style
    ax.annotate(cperiod,xy=(x2,y2),color = 'navy',ha='left',bbox=bbox_props)
    
    # Add legend (e.g., for "Best Track")
    legend = ax.legend(loc=1, shadow=True) # Location: upper right
    legend.get_frame().set_facecolor('w') 
  
    # --- Save and Show Plot ---
    output_filename = filename.rstrip('.dat')+'.png'
    plt.savefig(output_filename,dpi=100)
    print(f"Plot saved to {output_filename}")
    plt.show()
