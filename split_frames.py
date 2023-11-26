from PIL import Image
from moviepy.editor import *
import time
import numpy as np
import multiprocessing as mp
import argparse
import os

parser = argparse.ArgumentParser(description='Split Video into Frames')

parser.add_argument('--path',
                       type=str,
                       help='the path to video')

parser.add_argument('--rotate',
                       type=int,
                       help='rotation factor')


args = parser.parse_args()

input_file = args.path
# input_file = "E:\\streetLights\\Data\\29thDecember\\day\\left\\GH011366.MP4"

filename_with_extension = os.path.basename(input_file)

filename = filename_with_extension.split(".")[0]

oppath = os.path.dirname(input_file)

oppath = os.path.join(oppath, filename)

print("Output Path:", oppath)

# oppath = "E:\\streetLights\\Data\\29thDecember\\day\\left\\GH011366"

try:
    rotation = int(args.rotate)
except:
    rotation = None

print("Rotation : ", rotation)


isExist = os.path.exists(oppath)
if not isExist:
    os.makedirs(oppath)
    
print("File given : ", input_file)
print("Frames will be extracted at : ", oppath)


num_processes = mp.cpu_count()
# input_file = "../GH011369.MP4"
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

def process_video(group_number):
    
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
    
    start_time_sub = time.time()

    frames = subclip.iter_frames()
 
    # counter to count the frames
    counter = 0

    # using loop to transverse the frames
    for value in frames:

        # incrementing the counter
        counter += 1
        im = Image.fromarray(value)
        if rotation != None:
            im = im.rotate(rotation)
        im.save(oppath + "/%s.jpg" % str(frame_count_unit * group_number + counter))
        
#         print(oppath + "/%s.jpg" % str(frame_count_unit * group_number + counter), " process:", group_number, " frame count unit:", frame_count_unit)

    print("Done ", counter)
        
    t3_sub = time.time()
    print("Time taken to extract in process : ",group_number, "is ",  t3_sub - start_time_sub)
    return None


if __name__ == "__main__":
    kernel = np.ones((7, 7), np.float32) / 49
    start_time = time.time()
    p = mp.Pool(num_processes)
    p.map(process_video, range(num_processes))

    t3 = time.time()

    print("Total time taken to extract: ", t3 - start_time)
