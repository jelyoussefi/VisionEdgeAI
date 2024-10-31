import os, platform, subprocess
import fire
import cv2
import numpy as np
import time
import json
from threading import Thread, Condition
from collections import deque
import psutil
from time import perf_counter
from flask import Flask, redirect, url_for, render_template, make_response, jsonify, request, Response
from flask_bootstrap import Bootstrap
from flask_restful import Resource, Api, reqparse
from utils.yolov8_model  import YoloV8Model
from utils.images_capture import VideoCapture
import utils.perf_visualizer as pv

class ObjectDetector():
	def __init__(self, model, input, device="GPU", data_type="FP16"):

		self.app = Flask(__name__)
		self.port = 7000
		self.running = False 
		self.cv = Condition()
		self.cpu_loads = deque(maxlen=120)
		self.upload_folder = '/workspace/videos'  
		os.makedirs(self.upload_folder, exist_ok=True)  

		self.init(model, input, device, data_type)

	def init(self,  model, input, device="GPU", data_type="FP16"):
		self.cv.acquire()
		self.input = input
		self.model = YoloV8Model(model, device, data_type)
		self.frame = None
		self.cap = VideoCapture(input)
		self.frames_number = 0
		self.cpu_loads.clear()
		self.cpu_loads.append(psutil.cpu_percent(0.1))
		self.start_time = perf_counter()
		self.cv.release()

	def get_cpu_model(self):
		try:
			# First, try to get CPU information using platform.uname().processor or platform.processor()
			cpu_model = platform.uname().processor
			if not cpu_model or cpu_model == "x86_64":
				cpu_model = platform.processor()

			# Linux-specific method: Read /proc/cpuinfo for the model name
			if not cpu_model or cpu_model == "x86_64":
				if platform.system() == "Linux":
					with open("/proc/cpuinfo", "r") as f:
						for line in f:
							if "model name" in line:
								cpu_model = line.split(":")[1].strip()
								break

			return cpu_model or "CPU model not available"
		
		except Exception as e:
			print(f"Error fetching CPU model: {e}")
			return "CPU model not available"

	def get_gpu_model(self):
		try:
			gpu_model = "GPU model not available"
			
			result = subprocess.run(
				["lspci"], capture_output=True, text=True
			)
			for line in result.stdout.splitlines():
				if "VGA compatible controller" in line or "3D controller" in line:
					if "Intel" in line:
						gpu_model = line.split(": ")[1]
						break

			return gpu_model.replace("Intel Corporation", "").strip()
		
		except Exception as e:
			print(f"Error fetching GPU model: {e}")
			return "GPU model not available"
		
	def run(self):
		app = self.app

		Bootstrap(app)
		@app.route('/', methods=['GET', 'POST'])
		def home():
			files = os.listdir(self.upload_folder)
			files = [file for file in files if os.path.isfile(os.path.join(self.upload_folder, file))]
			default_file = files[0] if files else "No files available"
			cpu_model = self.get_cpu_model()
			gpu_model = self.get_gpu_model()

			return render_template('index.html', 
									default_device=self.model.device, 
									default_model=self.model.name,
									default_precision=self.model.data_type,
									default_file=default_file,
									cpu_model=cpu_model,
									gpu_model=gpu_model
								   )

		@app.route('/video_feed')
		def video_feed():
			return Response(self.video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

		@app.route('/upload', methods=['POST'])
		def upload_file():
			if 'file' not in request.files:
				return jsonify({"error": "No file part"}), 400
			file = request.files['file']
			if file.filename == '':
				return jsonify({"error": "No selected file"}), 400

			# Save the file to the upload folder
			file_path = os.path.join(self.upload_folder, file.filename)
			file.save(file_path)
			return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

		@self.app.route('/get_uploaded_files', methods=['GET'])
		def get_uploaded_files():
			files = os.listdir(self.upload_folder)
			files = [file for file in files if os.path.isfile(os.path.join(self.upload_folder, file))]
			return jsonify(files)

		@app.route('/select_source', methods=['POST'])
		def select_source():
			data = request.get_json()
			source = data.get('source')
			input = os.path.join(self.upload_folder, source)
			
			if not source:
				return jsonify({'error': 'No source provided'}), 400

			if input != self.input:
				print(f"Selected source: {input}")
				self.init(self.model.model_path, input, self.model.device, self.model.data_type)

			return jsonify({'message': f'Source {source} selected successfully'}), 200
	

		@app.route('/select_device', methods=['POST'])
		def select_device():
			data = request.get_json()
			device = data.get('device')

			if not device:
				return jsonify({'error': 'No device provided'}), 400

			# Perform any necessary action with the device selection
			print(f"Selected device: {device}")

			return jsonify({'message': f'Device {device} selected successfully'}), 200
		
		@app.route('/select_model', methods=['POST'])
		def select_model():
			data = request.get_json()
			model = data.get('model')
			if not model:
				return jsonify({'error': 'No model provided'}), 400
			if model != self.model.name:
				print(f"Selected model: {model}")
				self.init(model, self.input, self.model.device, self.model.data_type)

			return jsonify({'message': f'Model {model} selected successfully'}), 200
		

		@app.route('/select_precision', methods=['POST'])
		def select_precision():
			data = request.get_json()
			precision = data.get('precision')
			if not precision:
				return jsonify({'error': 'No precision provided'}), 400
			print(f"Selected precision: {precision}")
			return jsonify({'message': f'Precision {precision} selected successfully'}), 200

		self.frames_number = 0
		self.start_time = perf_counter()
		self.running = True

		self.proc = Thread(target=self.cpu_load_handler)
		self.proc.daemon = True
		self.proc.start()

		self.app.run(host='0.0.0.0', port=str(self.port), debug=False, threaded=True)

	def video_stream(self):

		self.cv.acquire()
		
		while self.running:

			self.cv.release()
			
			frame = self.cap.read()

			self.cv.acquire()

			if frame is not None:
				frame = self.model.predict(frame.copy())
				self.frames_number += 1
				pv.draw_perf(frame, self.model.device, self.fps(), self.cpu_load())

				ret, buffer = cv2.imencode('.jpg', frame)
				frame = buffer.tobytes()
				yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
				if not self.running:
					break
	

		self.cap = None
		self.cv.notify_all()
		self.cv.release()

	def fps(self):
		return self.frames_number/(perf_counter() - self.start_time)

	def cpu_load(self):
		return np.average(self.cpu_loads);

	def cpu_load_handler(self):

		self.cpu_loads.append(psutil.cpu_percent(0.1))
		
		while self.running:
			self.cv.acquire()
			self.cpu_loads.append(psutil.cpu_percent(0))
			self.cv.release()
			time.sleep(0.5)

def main( model, input, device, config, **kwargs ):
	app = ObjectDetector(model, input, device)
	app.run()

if __name__ == "__main__":
	fire.Fire(main)
