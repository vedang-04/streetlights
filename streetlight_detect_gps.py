import argparse
from pathlib import Path
from yolov5.detect import run

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
print(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt', help='model path(s)')
parser.add_argument('--source', type=str, default='../GH011369', help='file/dir/URL/glob, 0 for webcam')
parser.add_argument('--data', type=str, default=ROOT / 'yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_false', help='do not save images/videos')
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
parser.add_argument('--gps', action='store_false', help='save gps locations for streetlight detections')
parser.add_argument('--gps_file', type=str, default='../GH011369.csv', help='model path(s)')
parser.add_argument('--buffer_frames', default=5, type=int, help='buffer for streetlight tracking')
parser.add_argument('--superpoint_weights', default="./weights/sp_v6", type=str, help='weights for superpoint')
parser.add_argument('--k_best', type=int, default=10000,
                        help='Maximum number of keypoints to keep \
                        (default: 10000)')
opt = parser.parse_args()
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

if __name__ == "__main__":
    run(**vars(opt))