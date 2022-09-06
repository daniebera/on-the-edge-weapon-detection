"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import zmq
import pycuda.driver as cuda

from utils.yolo_with_plugins import TrtYOLO


#WINDOW_NAME = 'TrtYOLODemo'

#######################
# TCP based on ZeroMQ #
#######################

endpoint = "tcp://127.0.0.1:5004"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.CONFLATE, 1)
socket.bind(endpoint)


def recv_dict(socket, flags=0):
    """recv a json with data"""
    msg = socket.recv_json(flags=flags)
    return msg['data']


def parse_args():
    """Parse input arguments."""
    desc = ('Receive packets from first stage, and do'
            'real-time weapon detection with TensorRT optimized '
            'YOLO4-csp model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.5,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov4-csp]-[{dimension}], where '
              '{dimension} is a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args


def detect_packet(imgs, trt_yolo, conf_th):
    fps = 0.0
    tic = time.time()
    for img in imgs:
        boxes, confs, clss = trt_yolo.detect(img, conf_th)

        print('BOXES: ', len(boxes))
        res = {'boxes': boxes, 'confs': confs, 'clss': clss, 'thresh': conf_th}

        # Compute ExpDecaying Avg of fps
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc
        print('FPS: ', fps)
        return


def main():
    args = parse_args()
    if not os.path.isfile('./utils/%s.trt' % args.model):
        raise SystemExit('ERROR: file (./utils/%s.trt) not found!' % args.model)

    cuda.init()  # init pycuda driver
    cuda_ctx = cuda.Device(0).make_context()  # GPU 0

    trt_yolo = TrtYOLO(args.model)

    #####################################################
    # Receive img crops from people detection algorithm #
    #####################################################

    while True:
        packet = recv_dict(socket)

        if packet and packet != 'stop':
            detect_packet(packet, trt_yolo, args.conf_thresh)
        if packet == 'stop':
            print("Clean exit...")
            socket.close()
            context.term()
            print("Deallocating Inference Model.. ")
            del trt_yolo
            cuda_ctx.pop()
            del cuda_ctx
            print('Exit now..')
            exit(0)


if __name__ == '__main__':
    main()