import cv2
import numpy as np
from time import perf_counter

from utils.model import Model


class SSDModel(Model):
	def __init__(self, model_path, device, data_type, callback_function=None):
		super().__init__(model_path, device, data_type)
		self.user_callback = callback_function
		self.labels = [ "background", "person"]

	def callback(self, outputs, image):
		image = self.postprocess(outputs, image)
		if self.user_callback is not None:
			self.user_callback(image)

	

	def postprocess(self, detections, image, threshold=0.5):
		for _, label, score, xmin, ymin, xmax, ymax in detections[0][0]:
			if score >= threshold:
				xmin *= image.shape[1]
				xmax *= image.shape[1]
				ymin *= image.shape[0]
				ymax *= image.shape[0]
				self.plot_one_box(image, xmin, ymin, xmax, ymax, score, self.labels[int(label)])

		return image


	
