#!/usr/bin/env python
#--------------------------------------------------
#  目的:繪製GFS颱風路徑
#
#  陳登舜 製作  
#  版權所有翻印必究
#  2018-05-24
#
#  程式記錄:
#   2018-05-25  Deng-Shun  Created
#
#--------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sys
import pandas as pd
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
from datetime import datetime,timedelta

def get_data(filename) : 
  dtg_ini = sys.maxsize
  tau= []
  lat,lon, dtg = {}, {}, {}
  name, model = [], []
  
  f = open(filename,'r')
  header = f.readline().strip().split()
  
  for line in f:
    line = line.strip().split()

    if line[1] == 'number':
      dtg_ini = line[0]
      case  = line[4]
    elif line[0] = 'number':
      id = line[2]
      name = line[4]
    else:
                
  f.close()

  dtg_ini = dtg_ini  + 2000000000
  dtg_end = dtg_best + 2000000000

  return [big_lat, big_lon, big_dtg, best_lat, best_lon, best_dtg, dtg_ini, dtg_end, name, model]  

def plot_map(name,model,set_val=[80,0,190,50,120]): 
 
  # custom set_val
  #set_val=[80,0,190,50,120] 

  fig, ax = plt.subplots(figsize=(16,8))
  
  # Lambert Conformal Conic map.
  m = Basemap(llcrnrlon=set_val[0],llcrnrlat=set_val[1],urcrnrlon=set_val[2],urcrnrlat=set_val[3],lon_0=set_val[4],
              projection='lcc',lat_1=10.,lat_2=40.,
              resolution ='l',area_thresh=1000.,ax=ax)
  # draw coastlines, meridians and parallels.
  m.drawcoastlines()
  m.drawcountries()
  m.drawmapboundary(fill_color='white')
  m.fillcontinents(color='burlywood',lake_color='white')
  m.drawparallels(np.arange(15,70,20),labels=[1,1,0,0])
  m.drawmeridians(np.arange(80,190,20),labels=[0,0,0,1])
  
  twrftitle = 'CWB GFS('+model+') Typhoon Tracks - '+name
  plt.title(twrftitle)

  return fig, ax, m

def set_map(lat,lon,blat,blon):
  # find the max latitude and longitude
  indx = np.logical_and(lat > 0,dtg <= dtg_end) 
  lat_max = lat[indx].max()
  lat_min = lat[indx].min()
  lon_max = lon[indx].max()
  lon_min = lon[indx].min()
  if ( blat.max() > lat_max ) : lat_max = blat.max()
  if ( blat.min() < lat_min ) : lat_min = blat.min()
  if ( blon.max() > lon_max ) : lon_max = blon.max()
  if ( blon.min() < lon_min ) : lon_min = blon.min()
  
  print('lat max/min = ',lat_max,lat_min)
  print('lon max/min = ',lon_max,lon_min)

  # decide the map view 
  expendx = 5.
  expendy = 3.
  llcrnrlon = float(int(lon_min - expendx))
  llcrnrlat = float(int(lat_min - expendy))
  urcrnrlon = float(int(lon_max + expendx))
  urcrnrlat = float(int(lat_max + expendy))
  lon_0 = lon_min + 0.5*(lon_max-lon_min)

  return [llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,lon_0]

if __name__ == '__main__':

  # get arguments 
  parser = ArgumentParser(description = 'Plot CWB GFS track data',formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i','--infile',help='Input CWB GFS Track filename',type=str,nargs='+',required=True)
  args = parser.parse_args()
  filenames = args.infile
  
  for filename in filenames:
    # get data 
    print(filename)
    data = get_data(filename)
    if data[8] == []: continue 
   
    lat = np.asarray(data[0])
    lon = np.asarray(data[1]) 
    dtg = np.asarray(data[2]) 
    blat = np.asarray(data[3])
    blon = np.asarray(data[4]) 
    bdtg = np.asarray(data[5]) 
    
    dtg_ini = data[6] 
    dtg_end = data[7]
    name = data[8]
    model = data[9]
    
    set_val = set_map(lat,lon,blat,blon)
    
    # pre-define parameters
    print('ini/end = ',dtg_ini,dtg_end)
    print('Typhoon name = ',name)
    print('Model = ',model)
    print('map view : ',set_val[0],set_val[1],set_val[2],set_val[3])
    
    # plot map 
    fig, ax, m = plot_map(name,model,set_val)
    
    # plot fcst track
    lcolor = 'r'
    for icase in range(lat.shape[0]):
      indx = np.logical_and(lat[icase] > 0,dtg[icase] <= dtg_end)
      x,y = m(lon[icase][indx],lat[icase][indx])
      m.plot(x,y,color=lcolor)
    
    #plot best track
    lcolor = 'b'
    x,y = m(blon[0],blat[0])
    m.plot(x,y,color=lcolor,label='Best Track')
    
    # plot experiment period
    cperiod = str(dtg_ini)+' - '+str(dtg_end)
    x2,y2 = m(np.asarray(set_val[0])+1,np.asarray(set_val[1])+1.)
    bbox_props = dict(boxstyle="round",fc="w", ec="0.5", alpha=0.9)
    ax.annotate(cperiod,xy=(x2,y2),color = 'navy',ha='left',bbox=bbox_props)
    
    # save and show
    legend = ax.legend(loc=1, shadow=True)
    legend.get_frame().set_facecolor('w') 
  
    plt.savefig(filename.rstrip('.dat')+'.png',dpi=100)
    plt.show()
