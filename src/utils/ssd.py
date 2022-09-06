"""ssd.py

Implementation of TrtSSD class for people detection.
"""


import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda


def _preprocess_trt(img, shape=(300, 300)):
    """Preprocess image prior to TRT SSD inference."""
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img *= (2.0/255.0)
    img -= 1.0
    return img


def _postprocess_trt(img, out, conf_th, out_layout=7):
    """Postprocess TRT SSD output."""
    img_h, img_w, _ = img.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(out), out_layout):
        conf = float(out[prefix+2])
        cls = int(out[prefix + 1])
        if conf < conf_th or cls != 1:
            continue
        x1 = int(out[prefix+3] * img_w)
        y1 = int(out[prefix+4] * img_h)
        x2 = int(out[prefix+5] * img_w)
        y2 = int(out[prefix+6] * img_h)
        boxes.append((x1, y1, x2, y2))
        confs.append(conf)
        clss.append(cls)
    return boxes, confs, clss


class TrtSSD(object):
    """TrtSSD class encapsulates things needed to run SSD in TensorRT."""

    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        TRTbin = './utils/TRT_%s.bin' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:  # I/O bindings of the engine
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)  # linear piece of device memory
            bindings.append(int(cuda_mem))  # linear index into Context memory
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def __init__(self, model, input_shape):
        """Initialize TensorRT plugins, engine and context."""
        self.model = model
        self.input_shape = input_shape

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream

    def infer(self, img, conf_th=0.5):
        """Inference on the input image."""
        img_resized = _preprocess_trt(img, self.input_shape)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(
            self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(
            self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        out = self.host_outputs[0]
        return _postprocess_trt(img, out, conf_th)
