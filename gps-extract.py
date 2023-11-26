from gpsextraction.ffmpeg_utils import find_streams, find_number_frames
from gpsextraction.units import units
from gpsextraction.framemeta import framemeta_from
from gpsextraction.gpx import load_timeseries
from gpsextraction.framemeta_gpx import merge_gpx_with_gopro
from gpsextraction import timeseries_process
from gpsextraction.timeunits import timeunits
from gpsextraction.entry import Entry
from PIL import Image
from moviepy.editor import *
import time
import numpy as np
import multiprocessing as mp
import argparse
import os
import pandas as pd


num_processes = mp.cpu_count()

parser = argparse.ArgumentParser(description='Extract GPS')

parser.add_argument('--path',
                       type=str,
                       help='the path to video')

args = parser.parse_args()

input_file = args.path

cap = VideoFileClip(input_file)

start_time = 0

duration = cap.duration

frame_jump_unit = duration // num_processes
print("Each subprocess processes : ", frame_jump_unit, " seconds")

end_time = int(frame_jump_unit)

subclip = cap.subclip(start_time,end_time)

frame_count_unit = sum(1 for x in subclip.iter_frames())

print("Total number of frames : ", cap.reader.nframes)

fps = cap.fps

print("FPS: ", fps)

print("Duration: ", duration)

print("Each process handles : ", frame_count_unit)
filename_with_extension = os.path.basename(input_file)

filename = filename_with_extension.split(".")[0]

oppath = os.path.dirname(input_file)

print("Output Path:", oppath)


def process_video(group_number):

    df = pd.DataFrame()

    stream_info = find_streams(input_file)
    dimensions = stream_info.video.dimension

    print(f"Input file has size {dimensions}")

    gopro_frame_meta = framemeta_from(
        input_file,
        metameta=stream_info.meta,
        units=units
    )

    if len(gopro_frame_meta) < 1:
        raise IOError(
            f"Unable to load GoPro metadata from {input_file}. Use --debug-metadata to see more information")

    print(f"GoPro Timeseries has {len(gopro_frame_meta)} data points")

    find_number_frames(input_file)

    # gpx_timeseries = load_timeseries(gpx, units)
    # print(f"GPX Timeseries has {len(gpx_timeseries)} data points.. merging...")
    # merge_gpx_with_gopro(gpx_timeseries, gopro_frame_meta)

    gopro_frame_meta.process(timeseries_process.process_ses("point", lambda i: i.point, alpha=0.45))
    gopro_frame_meta.process_deltas(timeseries_process.calculate_speeds(), skip=18 * 3)
    gopro_frame_meta.process(timeseries_process.calculate_odo())
    gopro_frame_meta.process_deltas(timeseries_process.calculate_gradient(),
                                    skip=18 * 3)  # hack approx 18 frames/sec * 3 secs

    timelapse_correction = gopro_frame_meta.duration() / stream_info.video.duration

    print(f"Timelapse Factor = {timelapse_correction:.3f}")

    start_time = int(group_number * frame_jump_unit)
#     if group_number != num_processes - 1:
    end_time = int((group_number+1) * frame_jump_unit)
#     else:
#         end_time = duration
    if group_number + 1 == num_processes:
        end_time = duration
    
    subclip = cap.subclip(start_time,end_time)

    
    print("Starting at: ", start_time, " and ending at: ", end_time, " for process: ", group_number)
    
    total_frames = int(subclip.fps * subclip.duration)
    
    
    counter = 0

    # using loop to transverse the frames
    for i, (tstamp, frame) in enumerate(subclip.iter_frames(with_times=True)):
        
        tstamp = start_time + tstamp
        
#         print("Timestamp: ", tstamp, " process: ", group_number)

        e = gopro_frame_meta.get(timeunits(seconds= tstamp * timelapse_correction))
#         print("Lat :", e.__getattr__('point').lat, " Long: ", e.__getattr__('point').lon, " at: ", tstamp)

        df = df.append({'Frame ID': str(frame_count_unit * group_number + counter), 'Latitude': e.__getattr__('point').lat, 'Longitude':e.__getattr__('point').lon},
                       ignore_index=True)
        counter += 1

    print("Done i: ",i) 
    df.to_csv(str(group_number)+".csv", index=None, header=True, sep=',', encoding='utf-8',line_terminator="", float_format='%.16g')

    return None



def main():

    start_time = time.time()
    p = mp.Pool(num_processes)
    p.map(process_video, range(num_processes))

    s1 = pd.DataFrame()
    for i in range(num_processes):
        try:
            s2 = pd.read_csv(str(i)+".csv", encoding='utf-8')
        except:
            continue
        s1 = pd.concat([s1, s2], ignore_index=True)

        os.remove(str(i)+".csv")
        
    s1.to_csv(oppath+"/"+filename+".csv", index=None, header=True, sep=',', encoding='utf-8',line_terminator="", float_format='%.16g')
    t3 = time.time()

    print(t3 - start_time)


if __name__ == "__main__":

    main()