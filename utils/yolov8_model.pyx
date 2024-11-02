import os, argparse
import math
import ctypes
from typing import Tuple, Dict
from collections import deque
import cython
import numpy as np
import torch 

cimport numpy as np
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
						  int8_t, int16_t, int32_t, int64_t)
from libcpp cimport bool

from pathlib import Path
import shutil
import cv2
from PIL import Image
from time import perf_counter
import pathlib
from ultralytics import YOLO
from ultralytics.utils.plotting import colors


import openvino.runtime as ov
from openvino.runtime import Core, Model
import dpctl
import dpnp
	

np.import_array()


cdef class YoloV8ModelBase():
	def __init__(self, model_path, device, image_size=640, data_type="FP16"):
		self.device = device
		self.data_type = data_type
		self.model_path = model_path
		self.name = os.path.splitext(os.path.basename(model_path))[0] 
		model = YOLO(model_path)
		self.label_map = model.names

		model_name = os.path.basename(model_path).split('.')[0]
		model_path = os.path.join("./models", data_type, model_name + ".xml")

		if not os.path.isfile(model_path):
			half=True if data_type=="FP16" else False
			model.export(format="openvino", dynamic=False, half=half)
			dst_model_dir = os.path.dirname(model_path)
			src_model_dir = os.path.join("./", model_name+"_openvino_model")
			Path(dst_model_dir).mkdir(parents=True, exist_ok=True)
			for src_file in Path(src_model_dir).iterdir():
				if src_file.is_file():
					shutil.copy2(src_file, Path(dst_model_dir) / src_file.name)
			shutil.rmtree(src_model_dir)

		self.core = Core()
		self.ov_model = self.core.read_model(model_path)
		self.input_layer_ir = self.ov_model.input(0)
		self.input_height = self.input_layer_ir.shape[2]
		self.input_width = self.input_layer_ir.shape[3]
		self.ov_model.reshape({0: [1, 3, self.input_height, self.input_width]})
		self.model = self.core.compile_model(self.ov_model, self.device.upper())
		self.post_proc_device = dpctl.select_gpu_device()

		self.infer_times = []
		self.num_masks = 32
		self.conf_threshold = 0.5
		self.iou_threshold = 0.2


	def predict(self, image:np.ndarray):
		resized_image = self.preprocess(image)
		if self.device != "GPU":
			resized_image = resized_image.to(self.device)

		start_time = perf_counter()
		results = self.model(resized_image)

		self.infer_times.append((perf_counter() - start_time))
		boxes = results[0]
		if self.device != "GPU":
			boxes = boxes.cpu().numpy()
		masks = None #results[1].numpy() if len(results) > 1 else None
		boxes, scores, class_ids, mask_maps = self.postprocess(boxes, masks, image )
		
		image = self.draw_detections(image, boxes, scores, class_ids, 0.5, mask_maps=mask_maps)

		return image

	def fps(self):
		if len(self.infer_times) > 0:
			return 1/np.average(self.infer_times);
		else:
			return 0

	def preprocess(self, image):
		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		input_img = cv2.resize(input_img, (self.input_width, self.input_height))
		input_img = input_img / 255.0
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

		return torch.tensor(input_tensor)

	def postprocess(self, pred_boxes, pred_masks, orig_img):
		boxes, scores, class_ids, mask_pred = self.process_box_output(pred_boxes, orig_img)
		mask_maps = None
		try:
			if pred_masks is not None:
				mask_maps = self.process_mask_output(mask_pred, pred_masks, boxes, orig_img)
		except:
			pass
			
		return (boxes, scores, class_ids, mask_maps)

	cdef get_boxes(self, np.ndarray[float, ndim=2] box_predictions, np.ndarray[uint8_t, ndim=3] orig_img):
		cdef int img_height = orig_img.shape[0]
		cdef int img_width = orig_img.shape[1]

		cdef np.ndarray[float, ndim=2] boxes = box_predictions[:, :4]

		# Scale boxes to original image dimensions
		cdef np.ndarray[long, ndim=1] input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
		boxes = np.divide(boxes, input_shape, dtype=np.float32)
		boxes *= np.array([img_width, img_height, img_width, img_height])

		# Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
		cdef np.ndarray[float, ndim=2] xy = boxes.copy(); 
		boxes[..., 0] = xy[..., 0] - xy[..., 2] / 2
		boxes[..., 1] = xy[..., 1] - xy[..., 3] / 2
		boxes[..., 2] = xy[..., 0] + xy[..., 2] / 2
		boxes[..., 3] = xy[..., 1] + xy[..., 3] / 2
		
		# Check the boxes are within the image
		boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
		boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
		boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
		boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)

		return boxes

	cdef compute_iou(self, np.ndarray[float, ndim=1] box, np.ndarray[float, ndim=2] boxes):
		
		# Compute xmin, ymin, xmax, ymax for both boxes
		cdef np.ndarray[float, ndim=1] xmin = np.maximum(box[0], boxes[:, 0])
		cdef np.ndarray[float, ndim=1] ymin = np.maximum(box[1], boxes[:, 1])
		cdef np.ndarray[float, ndim=1] xmax = np.minimum(box[2], boxes[:, 2])
		cdef np.ndarray[float, ndim=1] ymax = np.minimum(box[3], boxes[:, 3])

		# Compute intersection area
		cdef np.ndarray[float, ndim=1] intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
		# Compute union area
		cdef float box_area = (box[2] - box[0]) * (box[3] - box[1])
		cdef np.ndarray[float, ndim=1] boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
		cdef np.ndarray[float, ndim=1] union_area = box_area + boxes_area - intersection_area

		# Compute IoU
		cdef np.ndarray[float, ndim=1]  iou = intersection_area / union_area

		return iou
	
	cdef nms(self, np.ndarray[float, ndim=2] boxes, np.ndarray[float, ndim=1] scores, float iou_threshold):
		# Sort by score
		cdef np.ndarray[long, ndim=1] sorted_indices = np.argsort(scores)[::-1]

		cdef list  keep_boxes = []
		cdef int box_id
		cdef np.ndarray[float, ndim=1] ious
		cdef np.ndarray[long, ndim=1] keep_indices

		while sorted_indices.size > 0:
			# Pick the last box
			box_id = sorted_indices[0]
			keep_boxes.append(box_id)

			# Compute IoU of the picked box with the rest
			ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

			# Remove boxes with IoU over the threshold
			keep_indices = np.where(ious < iou_threshold)[0]

			# print(keep_indices.shape, sorted_indices.shape)
			sorted_indices = sorted_indices[keep_indices + 1]

		return keep_boxes


	cdef process_box_output(self, np.ndarray[float, ndim=3] box_output, np.ndarray[uint8_t, ndim=3] orig_img):

		cdef np.ndarray[float, ndim=2]  predictions = np.squeeze(box_output).T

		cdef int num_classes = box_output.shape[1] - self.num_masks - 4
		# Filter out object confidence scores below threshold
		cdef np.ndarray[float, ndim=1]  scores = np.max(predictions[:, 4:4+num_classes], axis=1)
		predictions = predictions[scores > self.conf_threshold, :]
		scores = scores[scores > self.conf_threshold]

		if len(scores) == 0:
			return [], [], [], np.array([])

		cdef np.ndarray[float, ndim=2] box_predictions = predictions[..., :num_classes+4]
		cdef np.ndarray[float, ndim=2] mask_predictions = predictions[..., num_classes+4:]

		# Get the class with the highest confidence
		cdef np.ndarray[long, ndim=1] class_ids = np.argmax(box_predictions[:, 4:], axis=1)

		# Get bounding boxes for each object
		cdef np.ndarray[float, ndim=2] boxes = self.get_boxes(box_predictions, orig_img)

		# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
		cdef list indices = self.nms(boxes, scores, self.iou_threshold)

		return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

	def rescale_boxes(self, boxes, input_shape, image_shape):
		# Rescale boxes to original image dimensions
		input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
		boxes = np.divide(boxes, input_shape, dtype=np.float32)
		boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

		return boxes


	cdef process_mask_output(self,  mask_predictions,  mask_output, 
							  np.ndarray[float, ndim=2] boxes, np.ndarray[uint8_t, ndim=3] orig_img):

		if mask_predictions.shape[0] == 0:
			return []
		
		cdef int img_height = orig_img.shape[0]
		cdef int img_width = orig_img.shape[1]

		cdef np.ndarray[float, ndim=3] mask_output_ = np.squeeze(mask_output)

		# Calculate the mask maps for each box
		cdef int num_mask, mask_height, mask_width
		num_mask, mask_height, mask_width = (<object>mask_output_).shape  # CHW
		mask_output = mask_output.reshape((num_mask, -1))
		if self.post_proc_device is not None:
			mask_predictions = dpnp.array(mask_predictions, device=self.post_proc_device)
			mask_output =  dpnp.array(mask_output, device=self.post_proc_device)
		
		masks_ = self.sigmoid(mask_predictions @ mask_output).asnumpy()
		cdef np.ndarray[float, ndim=3] masks = masks_.reshape((-1, mask_height, mask_width))

		# Downscale the boxes to match the mask size
		scale_boxes = self.rescale_boxes(boxes, (img_height, img_width), (mask_height, mask_width))

		# For every box/mask pair, get the mask map
		mask_maps = np.zeros((len(scale_boxes), img_height, img_width))
		blur_size = (int(img_width / mask_width), int(img_height / mask_height))
		for i in range(len(scale_boxes)):

			scale_x1 = int(math.floor(scale_boxes[i][0]))
			scale_y1 = int(math.floor(scale_boxes[i][1]))
			scale_x2 = int(math.ceil(scale_boxes[i][2]))
			scale_y2 = int(math.ceil(scale_boxes[i][3]))

			x1 = int(math.floor(boxes[i][0]))
			y1 = int(math.floor(boxes[i][1]))
			x2 = int(math.ceil(boxes[i][2]))
			y2 = int(math.ceil(boxes[i][3]))

			scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
			crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

			crop_mask = cv2.blur(crop_mask, blur_size)

			crop_mask = (crop_mask > 0.5).astype(np.uint8)
			mask_maps[i, y1:y2, x1:x2] = crop_mask

		return mask_maps

	def sigmoid(self, x):
		return 1 / (1 + dpnp.exp(-x))

	

	def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
		img_height, img_width = image.shape[:2]
		size = min([img_height, img_width]) * 0.0008
		text_thickness = int(min([img_height, img_width]) * 0.002)

		if mask_maps is not None:
			mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)
		else:
			mask_img = image

		# Draw bounding boxes and labels of detections
		for box, score, class_id in zip(boxes, scores, class_ids):
			label = self.label_map[class_id]
				
			color = colors(class_id)

			x1, y1, x2, y2 = box.astype(int)

			# Draw rectangle
			cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

			caption = f'{label} {int(score * 100)}%'
			
			(tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
										  fontScale=size, thickness=text_thickness)
			th = int(th * 1.2)

			cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

			cv2.putText(mask_img, caption, (x1, y1),
						cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

		return mask_img


	def draw_masks(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
		mask_img = image.copy()

		# Draw bounding boxes and labels of detections
		for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
			color = colors(class_id)

			x1, y1, x2, y2 = box.astype(int)

			# Draw fill mask image
			if mask_maps is None:
				cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
			else:
				crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
				crop_mask_img = mask_img[y1:y2, x1:x2]
				crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
				mask_img[y1:y2, x1:x2] = crop_mask_img

		return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

class YoloV8Model(YoloV8ModelBase):
	def __init__(self, model_path, device, image_size=640):
		super().__init__(model_path, device, image_size)
		

	
