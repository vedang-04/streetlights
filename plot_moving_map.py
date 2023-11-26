import pandas as pd
import folium
import argparse
import os
from selenium import webdriver
import time

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

oppath = os.path.join(oppath, filename+"_map_points")

isExist = os.path.exists(oppath)
if not isExist:
    os.makedirs(oppath)
    

streetlight_locations_iter = pd.read_csv(location)

for ind, loc in streetlight_locations.iterrows():
    map = folium.Map(location=[loc["Latitude"], loc["Longitude"]],tiles='cartodbpositron', zoom_start =18, max_zoom=30, control_scale=True)
    if loc['count'] == 2:
        folium.Marker([loc["Latitude"], loc["Longitude"]], popup=folium.Popup(str(loc["Frame ID"]), show=True),
                      icon=folium.Icon(color='blue', icon_color='#FFFF00')).add_to(map)
    else:
        folium.Marker([loc["Latitude"], loc["Longitude"]], popup=folium.Popup(str(loc["Frame ID"]), show=True), icon=folium.Icon(color='red')).add_to(map)


    for index, location_info in streetlight_locations_iter.iterrows():
        if ind == index:
            continue
        if location_info['count'] == 2:
            folium.Marker([location_info["Latitude"], location_info["Longitude"]], popup=location_info["Frame ID"],
                        icon=folium.Icon(color='blue', icon_color='#FFFF00')).add_to(map)
        else:
            folium.Marker([location_info["Latitude"], location_info["Longitude"]], popup=location_info["Frame ID"], icon=folium.Icon(color='red')).add_to(map)

    filepath = oppath + "/" + filename+"_"+str(loc["Frame ID"]) + "_map.html"
    map.save(filepath)


    delay=5

    # Download chromedriver from https://sites.google.com/chromium.org/driver/?pli=1
    
    fn=filename+"_"+str(loc["Frame ID"]) + "_map.html"
    tmpurl='file://{path}/{mapfile}'.format(path=oppath,mapfile=fn)

    browser = webdriver.Chrome("D:\\Downloads\\chromedriver_win32\\chromedriver.exe")
    browser.get(tmpurl)

    #Give the map tiles some time to load
    time.sleep(delay)
    browser.save_screenshot(oppath + "/" + filename+"_"+str(loc["Frame ID"]) + "_map.png")
    browser.quit()
    
    #remove html files
    os.remove(filepath)