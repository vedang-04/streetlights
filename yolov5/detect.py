# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
# from msilib.schema import Condition
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from scipy import stats as st
import tensorflow as tf  # noqa: E402


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import numpy as np
from utils.superpoint import *
import math

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        gps = False,
        gps_file = None,
        buffer_frames = 5,
        superpoint_weights = None,
        k_best = 10000
):
    if gps:
        gps_df = pd.read_csv(gps_file)
        print(gps_df.info())
        gps_df_write = pd.DataFrame()
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    frame_box = {}

    frame_data = {}

    frame_counter_prev = 0
    frame_counter_current = 0

    # conditions = -1
    for path, im, im0s, vid_cap, s in dataset:
        frame_counter_current +=1
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            M = imc.shape[1] // 4
            x0 = M
            x1 = M+M+M
            # x2 = M+M+M

            N = imc.shape[0] // 4

            shapes = np.zeros_like(imc, np.uint8)
            cv2.rectangle(shapes, (M, imc.shape[0]), (M+M, 0), (255, 255, 255), cv2.FILLED)
            alpha = 0.5
            mask = shapes.astype(bool)
            im0[mask] = cv2.addWeighted(im0, alpha, shapes, 1 - alpha, 0)[mask]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # print(torch.tensor(xyxy).numpy()[0])
                    # print(x0)
                    # print(x1)
                    # print(torch.tensor(xyxy).numpy())
                    # print(3 * N)
                    # print(N)
                    if not(torch.tensor(xyxy).numpy()[0] >= x0 and torch.tensor(xyxy).numpy()[0] <=x1 and torch.tensor(xyxy).numpy()[1] <= 3 * N):
                        continue
                    # print(xyxy)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if gps:                    

                        frame_id = Path(path).stem.split('_')[0]

                        weights_dir = Path(superpoint_weights, "sp_v6")

                        print(frame_box)

                        print("frameid")
                        print(frame_id)

                        if len(frame_box) == 0:

                            frame_box[frame_id] = [torch.tensor(xyxy).numpy()]


                            img = cv2.resize(imc, (416, 416))
                            
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img_preprocessed = img
                            frame_data[frame_id] = img_preprocessed
                            # conditions = 1 # No frame with streetlight uptil now

                        elif frame_id in frame_box.keys():
                            boxes = frame_box[frame_id]
                            boxes.append([torch.tensor(xyxy).numpy()])
                            frame_box[frame_id] = boxes
                            img = cv2.resize(imc, (416, 416))
                            
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img_preprocessed = img
                            frame_data[frame_id] = img_preprocessed
                            # conditions = 1  # Same Frame with Multiple Lights

                        elif frame_id not in frame_box.keys():

                            # Buffer Frames. If current frame id is more than 5 frames later, then new streetlight

                            # if float(frame_id) > float(list(frame_box.keys())[-1])+ buffer_frames:
                            print("frame counter")
                            print(frame_counter_current)
                            print(frame_counter_prev)
                            if frame_counter_current > frame_counter_prev + buffer_frames:
                                
                                lats = []
                                longs = []
                                for frame_id_box in frame_box.keys():
                                    print("frame id box : ", frame_id_box)
                                   
                                    loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]
                                    print("Loc : ", loc)
                                    lats.append(loc.iloc[0]['Latitude'])
                                    longs.append(loc.iloc[0]['Longitude'])

                                avg_lat = np.mean(lats)
                                avg_long = np.mean(longs)

                                loc = gps_df.loc[gps_df['Frame ID'] == int(list(frame_box.keys())[-1])]

                                flag = 0
                                for frame_id_box in frame_box.keys():
                                    if len(frame_box[frame_id_box]) == 2:
                                        flag = 1
                                        break

                                count_box = 0

                                if flag == 1:
                                    count_box = 2
                                else:
                                    count_box = 1

                                
                                loc['count'] = count_box
                                loc.iloc[0]['Latitude'] = avg_lat
                                loc.iloc[0]['Longitude'] = avg_long
                                temp = pd.concat([gps_df_write, loc], ignore_index=True)

                                gps_df_write.drop(gps_df_write.index[0:], inplace=True)

                                gps_df_write[temp.columns] = temp

                                
                                frame_box = {}
                                frame_data = {}
                                frame_box[frame_id] = [torch.tensor(xyxy).numpy()]
                                frame_data[frame_id] = img_preprocessed
                                print("frameid1")
                                print(frame_id)
                                print(frame_box)
                                print(gps_df_write)
                                # conditions = 1 # New Frame with Streetlight
                            else:
                                
                                img = cv2.resize(imc, (416, 416))
                            
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                img_preprocessed = img

                                

                                img2, cb2 = band_func(img_preprocessed)

                                
                                img1 = frame_data[list(frame_data.keys())[-1]]

                                img1, cb1 = band_func(img1)

                                graph = tf.Graph()
                                with tf.compat.v1.Session(graph=graph) as sess:
                                    tf.compat.v1.saved_model.loader.load(sess,
                                                                            [tf.compat.v1.saved_model.tag_constants.SERVING],
                                                                            str(weights_dir))

                                    input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
                                    output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
                                    output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

                                    out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                                    feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
                                    keypoint_map1 = np.squeeze(out1[0])
                                    descriptor_map1 = np.squeeze(out1[1])

                                    
                                    
                                    kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
                                    keypoint_map1, descriptor_map1, cb1, preprocess_coord(frame_box[list(frame_box.keys())[-1]], img1), img1, k_best)

                                    if kp1 == None:
                                        lats = []
                                        longs = []
                                        for frame_id_box in frame_box.keys():
                                            loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]
                                            lats.append(loc.iloc[0]['Latitude'])
                                            longs.append(loc.iloc[0]['Longitude'])

                                        avg_lat = np.mean(lats)
                                        avg_long = np.mean(longs)

                                        loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]

                                        flag = 0
                                        for frame_id_box in frame_box.keys():
                                            if len(frame_box[frame_id_box]) == 2:
                                                flag = 1
                                                break

                                        count_box = 0

                                        if flag == 1:
                                            count_box = 2
                                        else:
                                            count_box = 1
                                        
                                        loc['count'] = count_box
                                        loc.iloc[0]['Latitude'] = avg_lat
                                        loc.iloc[0]['Longitude'] = avg_long
                                        temp = pd.concat([gps_df_write, loc], ignore_index=True)

                                        gps_df_write.drop(gps_df_write.index[0:], inplace=True)

                                        gps_df_write[temp.columns] = temp

                                        
                                        frame_box = {}
                                        frame_data = {}
                                        frame_box[frame_id] = [torch.tensor(xyxy).numpy()]
                                        frame_data[frame_id] = img_preprocessed

                                    else:


                                        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                                            feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
                                        keypoint_map2 = np.squeeze(out2[0])
                                        descriptor_map2 = np.squeeze(out2[1])


                                        kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
                                            keypoint_map2, descriptor_map2, cb2, preprocess_coord([(xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()], img2), img2, k_best)

                                        if kp2 == None:
                                            lats = []
                                            longs = []
                                            for frame_id_box in frame_box.keys():
                                                loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]
                                                lats.append(loc.iloc[0]['Latitude'])
                                                longs.append(loc.iloc[0]['Longitude'])

                                            avg_lat = np.mean(lats)
                                            avg_long = np.mean(longs)

                                            loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]

                                            flag = 0
                                            for frame_id_box in frame_box.keys():
                                                if len(frame_box[frame_id_box]) == 2:
                                                    flag = 1
                                                    break

                                            count_box = 0

                                            if flag == 1:
                                                count_box = 2
                                            else:
                                                count_box = 1

                                            
                                            loc['count'] = count_box
                                            loc.iloc[0]['Latitude'] = avg_lat
                                            loc.iloc[0]['Longitude'] = avg_long
                                            temp = pd.concat([gps_df_write, loc], ignore_index=True)

                                            gps_df_write.drop(gps_df_write.index[0:], inplace=True)

                                            gps_df_write[temp.columns] = temp

                                            
                                            frame_box = {}
                                            frame_data = {}
                                            frame_box[frame_id] = [torch.tensor(xyxy).numpy()]
                                            frame_data[frame_id] = img_preprocessed

                                        else:

                                            min_kp = min(len(kp1), len(kp2))
                                            kp1 = kp1[:min_kp]
                                            kp2 = kp2[:min_kp]
                                            desc1 = desc1[:min_kp]
                                            desc2 = desc2[:min_kp]

                                            m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
                                            print("matches")
                                            print(len(matches))
                                            # if (len(m_kp1)<4)|(len(m_kp2)<4)|
                                            if (len(matches)<5):
                                                lats = []
                                                longs = []
                                                for frame_id_box in frame_box.keys():
                                                    loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]
                                                    lats.append(loc.iloc[0]['Latitude'])
                                                    longs.append(loc.iloc[0]['Longitude'])

                                                avg_lat = np.mean(lats)
                                                avg_long = np.mean(longs)

                                                loc = gps_df.loc[gps_df['Frame ID'] == int(frame_id_box)]

                                                flag = 0
                                                for frame_id_box in frame_box.keys():
                                                    if len(frame_box[frame_id_box]) == 2:
                                                        flag = 1
                                                        break

                                                count_box = 0

                                                if flag == 1:
                                                    count_box = 2
                                                else:
                                                    count_box = 1

                                                
                                                loc['count'] = count_box
                                                loc.iloc[0]['Latitude'] = avg_lat
                                                loc.iloc[0]['Longitude'] = avg_long
                                                temp = pd.concat([gps_df_write, loc], ignore_index=True)

                                                gps_df_write.drop(gps_df_write.index[0:], inplace=True)

                                                gps_df_write[temp.columns] = temp

                                                
                                                frame_box = {}
                                                frame_data = {}
                                                frame_box[frame_id] = [torch.tensor(xyxy).numpy()]
                                                frame_data[frame_id] = img_preprocessed
                                            else:
                                                frame_box[frame_id] = [torch.tensor(xyxy).numpy()]
                                                frame_data[frame_id] = img_preprocessed


                    frame_counter_prev = frame_counter_current
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if gps:
        lats = []
        longs = []
        for frame_id_box in frame_box.keys():
            loc = gps_df.loc[gps_df['Frame ID'] == float(frame_id_box)]
            lats.append(loc.iloc[0]['Latitude'])
            longs.append(loc.iloc[0]['Longitude'])

        avg_lat = np.mean(lats)
        avg_long = np.mean(longs)

        loc = gps_df.loc[gps_df['Frame ID'] == int(list(frame_box.keys())[-1])]

        flag = 0
        for frame_id_box in frame_box.keys():
            if len(frame_box[frame_id_box]) == 2:
                flag = 1
                break

        count_box = 0

        if flag == 1:
            count_box = 2
        else:
            count_box = 1

        
        loc['count'] = count_box
        loc.iloc[0]['Latitude'] = avg_lat
        loc.iloc[0]['Longitude'] = avg_long
        temp = pd.concat([gps_df_write, loc], ignore_index=True)

        gps_df_write.drop(gps_df_write.index[0:], inplace=True)

        gps_df_write[temp.columns] = temp
        
        filename_with_extension = os.path.basename(gps_file)

        filename = filename_with_extension.split(".")[0]

        oppath = os.path.dirname(gps_file)

        print("Output Path:", oppath)


        gps_df_write.to_csv(oppath+"/" + filename + "_streetlights.csv", index=None, header=True, sep=',', encoding='utf-8',line_terminator="")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--gps', action='store_true', help='save gps locations for streetlight detections')
    parser.add_argument('--gps_file', type=str, default='../GH011369.csv', help='model path(s)')
    parser.add_argument('--buffer_frames', default=5, type=int, help='buffer for streetlight tracking')
    parser.add_argument('--superpoint_weights', default="./weights/sp_v6", type=str, help='weights for superpoint')
    parser.add_argument('--k_best', type=int, default=10000,
                            help='Maximum number of keypoints to keep \
                            (default: 10000)')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
