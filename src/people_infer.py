import time
import argparse

import cv2
import zmq
import pycuda.driver as cuda

from utils.camera import add_camera_args, Camera
from utils.ssd import TrtSSD
from utils.postprocess import compute_img_crops, Box

INPUT_PDET = (300, 300)
INPUT_DOBJDET = (416, 416)
SUPPORTED_MODELS = [
    'ssdlite_mobilenet_v2_coco',
]

#######################
# TCP based on ZeroMQ #
#######################

endpoint = "tcp://127.0.0.1:5004"

context = zmq.Context()

socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.CONFLATE, 1)
socket.connect(endpoint)

try:
    socket.setsockopt_string(zmq.SUBSCRIBE, b"")
except:
    socket.setsockopt(zmq.SUBSCRIBE, b"")


def send_dict(socket, A, flags=0):
    """send a dict with data"""
    md = dict(data = A)
    return socket.send_json(md, flags)


def parse_args():
    """Parse input arguments."""
    desc = ('Do real-time people detection with TensorRT'
            'SSD engine on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('-m', '--model', type=str,
                        default='ssdlite_mobilenet_v2_coco',
                        choices=SUPPORTED_MODELS)
    parser.add_argument('-t', '--thresh', type=float,
                        default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    conf_th = args.thresh

    cuda.init()  # init pycuda driver
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0
    trt_ssd = TrtSSD(args.model, INPUT_PDET)

    fps = 0.0
    tic = time.time()
    while True:
        #  Wait for next request from client
        img = cam.read()
        if img is None:
            send_dict(socket, 'stop')
            print("Clean exit...")
            socket.close()
            context.term()
            print("Deallocating Inference Model.. ")
            del trt_ssd
            cuda_ctx.pop()
            del cuda_ctx
            print('Exit now..')
            exit(0)
        boxes, confs, clss = trt_ssd.infer(img, conf_th)

        print('BOXES: ', len(boxes))
        res = {'boxes': boxes, 'confs': confs, 'clss': clss, 'thresh': conf_th}

        #########################################
        # Generate 416x416 img crops on people) #
        #########################################

        img_h, img_w, _ = img.shape
        imgs = []
        for bb in res['boxes']:
            bb = Box(bb)
            max_dim = max(INPUT_DOBJDET[0], max(bb.w, bb.h))
            xstart, xend, ystart, yend = compute_img_crops(img_w, img_h, max_dim, bb.xcen, bb.ycen)
            img_new_w, img_new_h = xend - xstart, yend - ystart
            print('Crop H,W:', img_new_h, img_new_w)
            img_new = img[ystart:yend, xstart:xend, :]  # img has shape [H,W,C]
            ratio = INPUT_DOBJDET[0] / img_new.shape[0]
            if ratio != 1:
                # resize img_new according to ratio
                img_new = cv2.resize(img_new, INPUT_DOBJDET)
                print('Resize H,W', img_new.shape[0], img_new.shape[1])
            imgs.append(img_new)

        send_dict(socket, imgs)

        # Compute ExpDecaying Avg of fps
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc
        print('FPS: ', fps)


if __name__ == '__main__':
    main()