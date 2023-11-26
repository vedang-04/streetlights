import pandas as pd
import numpy as np
import folium
import argparse
import os

def distance(s_lat, s_lng, e_lat, e_lng):
    
    # approximate radius of earth in km
    R = 6373.0*1000
    
    s_lat = s_lat*np.pi/180.0                      
    s_lng = np.deg2rad(s_lng)     
    e_lat = np.deg2rad(e_lat)                       
    e_lng = np.deg2rad(e_lng)  
    
    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
    
    return 2 * R * np.arcsin(np.sqrt(d)) 


def display(mymap, points, distance, tree_count):
  thresh1 = 2
  thresh2 = 4
  thresh3 = 6
  thresh4 = 8
  print("Thresholds: ", thresh1, thresh2, thresh3, thresh4)
  print("Tree Count: ", tree_count)
  print(points[0])
  print(points[-1])
  # folium.Marker([points[0][0], 
  #                  points[0][1]],
                  
  #                icon = folium.Icon(color='green',icon='plus')).add_to(mymap)
  if tree_count < thresh1:
    print("1")
    folium.PolyLine(points, color='black', weight=4.5, opacity=.9).add_to(mymap) 
  elif tree_count < thresh2:
    print("2")
    folium.PolyLine(points, color='#ed3833', weight=4.5, opacity=.9).add_to(mymap) 
  elif tree_count < thresh3:
    print("3")
    folium.PolyLine(points, color='#2070c0', weight=4.5, opacity=.9).add_to(mymap)
  elif tree_count < thresh4:
    print("4")
    folium.PolyLine(points, color='#70ed3e', weight=4.5, opacity=.9).add_to(mymap)
  else:
    print("5")
    folium.PolyLine(points, color='#548236', weight=4.5, opacity=.9).add_to(mymap)

parser = argparse.ArgumentParser(description='Density Plots')

parser.add_argument('--gps',
                       type=str,
                       help='the path to gps locations of video')

parser.add_argument('--streetlights',
                       type=str,
                       help='the path to gps locations of streetlights')

args = parser.parse_args()

gps_file = args.gps
street_gps = args.streetlights

gps_df = pd.read_csv(gps_file)

streetlights_gps = pd.read_csv(street_gps)

filename_with_extension = os.path.basename(gps_file)

filename = filename_with_extension.split(".")[0]

oppath = os.path.dirname(gps_file)


prev_lat = -99999
prev_long = -99999

tot = 0

init_lat = -9999
init_long = -9999

first_lat = -9999
first_long = -9999

mymap = folium.Map( location=[gps_df.Latitude.mean(), gps_df.Longitude.mean()], zoom_start=13, tiles=None)
folium.TileLayer('cartodbpositron').add_to(mymap)
points = []

for index, row in gps_df.iterrows():
    # print(prev_lat)
    if (float(row['Latitude']), float(row['Longitude'])) == (first_lat,first_long):
      continue
    # print(index)
    if prev_lat == -99999:
        prev_lat, prev_long = float(row['Latitude']), float(row['Longitude'])
        points.append((float(row['Latitude']), float(row['Longitude'])))
        init_lat, init_long = float(row['Latitude']), float(row['Longitude'])
        first_lat, first_long = float(row['Latitude']), float(row['Longitude'])
        continue
    latitude, longitude = float(row['Latitude']), float(row['Longitude'])

    tot += distance(prev_lat, prev_long, latitude, longitude)

    points.append((float(row['Latitude']), float(row['Longitude'])))

    if tot > 100:
        distances = tot
        tot = 0

        ind_list = []

        dist = distance(init_lat, init_long, latitude, longitude)

        

        for index, street_row in streetlights_gps.iterrows():

            s_dist_i = distance(init_lat, init_long, float(street_row['Latitude']), float(street_row['Longitude']))
            s_dist_e = distance(init_lat, init_long, float(street_row['Latitude']), float(street_row['Longitude']))

            if s_dist_i+s_dist_e < dist:
                ind_list.append(index)
        
        count = len(ind_list)

        for ind in ind_list:
            streetlights_gps = streetlights_gps.drop(ind)

        # print(streetlights_gps)

        display(mymap, points, distances,count)

        points = [points[-1]]


        init_lat, init_long = float(row['Latitude']), float(row['Longitude'])


    prev_lat, prev_long = float(row['Latitude']), float(row['Longitude'])

filepath = oppath+ "/" + filename+ '_map_density.html'
mymap.save(filepath)
