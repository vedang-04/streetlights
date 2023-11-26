import pandas as pd
import folium
import argparse
import os


parser = argparse.ArgumentParser(description='Plot GPS')

parser.add_argument('--path',
                       type=str,
                       help='the path to data')

args = parser.parse_args()

location = args.path

streetlight_locations = pd.read_csv(location)

filename_with_extension = os.path.basename(location)

filename = filename_with_extension.split(".")[0]

oppath = os.path.dirname(location)

# Creating the map and adding points to it
map = folium.Map(location=[streetlight_locations.Latitude.mean(), streetlight_locations.Longitude.mean()],tiles='cartodbpositron', zoom_start =18, max_zoom=30, control_scale=True)

for index, location_info in streetlight_locations.iterrows():
    if location_info['count'] == 2:
        folium.Marker([location_info["Latitude"], location_info["Longitude"]], popup=location_info["Frame ID"],
                      icon=folium.Icon(color='blue', icon_color='#FFFF00')).add_to(map)
    else:
        folium.Marker([location_info["Latitude"], location_info["Longitude"]], popup=location_info["Frame ID"], icon=folium.Icon(color='red')).add_to(map)

filepath = oppath + "/" + filename + "_map.html"
map.save(filepath)