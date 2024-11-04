import os
from collections import deque
from threading import Condition
import cv2
import numpy as np
from time import perf_counter
from openvino.runtime import Core, AsyncInferQueue


class Model():
	def __init__(self, model_path, device, data_type, preprocess, call_back):
		
		self.cv = Condition()
		self.model_path = model_path
		self.device = device
		self.data_type = data_type
		self.callback_function = call_back
		self.request_queue_size = 2
		self.latencies = deque(maxlen=10)
		if preprocess is not None:
			self.preprocess = preprocess

		self.core = Core()
		
		if model_path is not None:
			self.init(model_path)

	def init(self, model_path):

		self.model_path = model_path
		self.ov_model = self.core.read_model(model_path)
		self.input_layer_ir = self.ov_model.input(0)
		self.input_layer_name = self.input_layer_ir.get_any_name()
		self.input_height = self.input_layer_ir.shape[2]
		self.input_width = self.input_layer_ir.shape[3]		
		self.ov_model.reshape({0: [1, 3, self.input_height, self.input_width]})
		self.model = self.core.compile_model(self.ov_model, self.device.upper())
		self.output_tensor = self.model.outputs[0]

		self.infer_queue = AsyncInferQueue(self.model, self.request_queue_size)
		self.infer_queue.set_callback(self.callback)
		self.frames_number = 0
		self.start_time = None


	def predict(self, image:np.ndarray):

		self.cv.acquire()

		if 	self.frames_number == 0:
			self.start_time = perf_counter()

		self.frames_number += 1

		self.cv.release()

		resized_image = self.preprocess(image)
		
		start_time = perf_counter()
		self.infer_queue.start_async(inputs={self.input_layer_name: resized_image}, userdata=(image, start_time))


	def callback(self, infer_request, userdata):
		image, start_time = userdata

		self.latencies.append((perf_counter() - start_time))

		result = infer_request.results[self.output_tensor] 

		if self.callback_function is not None:
			self.callback_function(result, image)			

	def fps(self):
		return self.frames_number/(perf_counter() - self.start_time) if self.start_time is not None else 0


	def latency(self):
		if len(self.latencies) > 0:
			return 1000*np.mean(self.latencies)
		else:
			return 0

	def preprocess(self, image):
		return None


	