"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM or USB webcam.
"""


import logging
import threading
import subprocess

import numpy as np
import cv2


# The following flag ise used to control whether to use a GStreamer
# pipeline to open USB webcam source.  If set to False, we just open
# the webcam using cv2.VideoCapture(index) machinery. i.e. relying
# on cv2's built-in function to capture images from the webcam.
USB_GSTREAMER = True


def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')
    parser.add_argument('--usb', type=int, default=None,
                        help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--width', type=int, default=640,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=480,
                        help='image height [480]')
    return parser


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            break
    cam.thread_running = False


class Camera():
    """Camera class which supports reading images from theses video sources:

    1. RTSP (IP CAM)
    2. USB webcam
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        self._open()  # try to open the camera

    def _open(self):
        """Open camera based on command line arguments."""
        if self.cap is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args
        if a.rtsp:
            logging.info('Camera: using RTSP stream %s' % a.rtsp)
            self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
            self._start()
        elif a.usb is not None:
            logging.info('Camera: using USB webcam /dev/video%d' % a.usb)
            self.cap = open_cam_usb(a.usb, a.width, a.height)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

    def isOpened(self):
        return self.is_opened

    def _start(self):
        if not self.cap.isOpened():
            logging.warning('Camera: starting while cap is not opened!')
            return

        # Grab an img to determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image!')
            self.is_opened = False
            return
        self.is_opened = True
        self.img_height, self.img_width, _ = self.img_handle.shape
        # start a child thread to grab img from rtsp or usb
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False

    def read(self):
        """Read a frame from the camera object.

        Returns None if the camera runs out of image or error.
        """
        if not self.is_opened:
            return None
        else:
            return self.img_handle

    def release(self):
        self._stop()
        try:
            self.cap.release()
        except:
            pass
        self.is_opened = False

    def __del__(self):
        self.release()
