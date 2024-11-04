import os, time, platform, queue, psutil, subprocess, random, logging
import fire
import cv2
import numpy as np
import csv
import io
from threading import Condition
from collections import deque
from time import perf_counter
from flask import Flask, render_template, jsonify, request, Response
from flask_bootstrap import Bootstrap
from utils.yolov8_model import YoloV8Model
from utils.images_capture import VideoCapture

# Disable Flask's default request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set level to ERROR to hide access logs

class ObjectDetector:
	def __init__(self):
		self.app = Flask(__name__)
		self.port = 5000
		self.running = False
		self.cv = Condition()
		self.queue = queue.Queue(maxsize=4)  
		self.upload_folder = '/workspace/videos'
		self.start_time = perf_counter()
		os.makedirs(self.upload_folder, exist_ok=True)
		self.model = self.model_path = self.device = self.input = self.data_type = self.cap = None

	def init(self, model_path, input, device="GPU", data_type="FP16"):
		self.cv.acquire()

		if (model_path != self.model_path) or (device != self.device) or (data_type != self.data_type):
			self.model_path = model_path
			self.device = device
			self.data_type = data_type
			if model_path is not None and device is not None and data_type is not None:
				self.model = YoloV8Model(model_path, device, data_type, self.callback_function)
		self.frame = None
		if input != self.input:
			self.input = input
			self.cap = VideoCapture(input)
		self.cv.release()

	def callback_function(self, frame):
		#self.cv.acquire()
		self.queue.put(frame)
		#self.cv.notify_all()
		#self.cv.release()

	def get_cpu_model(self):
		try:
			cpu_model = platform.uname().processor or platform.processor()
			if not cpu_model or cpu_model == "x86_64" and platform.system() == "Linux":
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

	def get_power_consumption(self):

		total_energy = -1

		command = ['pcm', '/csv', '0.5', '-nc', '-i=1', '-ns']
		
		try:
			result = subprocess.run(command, capture_output=True, text=True, timeout=60)
			output = result.stdout

			# Parse the CSV output
			csv_reader = csv.reader(io.StringIO(output))
			next(csv_reader, None)  # Skip the first line if it's a header

			header_row = next(csv_reader, None)
			if header_row:
				# Find the indices of the energy columns
				try:
					proc_energy_index = header_row.index("Proc Energy (Joules)")
					power_plane_0_index = header_row.index("Power Plane 0 Energy (Joules)")
					power_plane_1_index = header_row.index("Power Plane 1 Energy (Joules)")
				except ValueError:
					proc_energy_index = power_plane_0_index = power_plane_1_index = None

			# Read the data row with actual values
			data_row = next(csv_reader, None)
			if data_row:
				# Retrieve energy values if the indices are available and add them to the total
				proc_energy = float(data_row[proc_energy_index]) if proc_energy_index is not None else 0.0
				power_plane_0 = float(data_row[power_plane_0_index]) if power_plane_0_index is not None else 0.0
				power_plane_1 = float(data_row[power_plane_1_index]) if power_plane_1_index is not None else 0.0

				# Add the current readings to the cumulative total
				total_energy += proc_energy + power_plane_0 + power_plane_1

			total_energy

		except Exception as e:
			pass
		
		return total_energy

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

			return render_template('index.html', default_device="GPU", default_model="yolov8n",
								   default_precision="FP16", default_file=default_file,
								   cpu_model=cpu_model, gpu_model=gpu_model)

		@app.route('/video_feed')
		def video_feed():
			return Response(self.video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

		@app.route('/metrics', methods=['GET'])
		def get_metrics():
			try:
				cpu_percent = psutil.cpu_percent(interval=None)
				power_data = self.get_power_consumption()
				fps = 0  
				latency = 0
				if self.model is not None:
					fps = int(self.model.fps())
					latency = int(self.model.latency())

				metrics = {
					'cpu_percent': cpu_percent,
					'power_data': power_data,
					'fps': fps,
					'latency': latency
				}
				
				return jsonify(metrics)
			except Exception as e:
				print("Error gathering metrics:", e)
				return jsonify({'error': 'Failed to gather metrics'}), 500

		@app.route('/upload', methods=['POST'])
		def upload_file():
			if 'file' not in request.files:
				return jsonify({"error": "No file part"}), 400
			file = request.files['file']
			if file.filename == '':
				return jsonify({"error": "No selected file"}), 400

			file_path = os.path.join(self.upload_folder, file.filename)
			file.save(file_path)
			return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

		@app.route('/get_uploaded_files', methods=['GET'])
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

			self.init(self.model_path, input, self.device, self.data_type)

			return jsonify({'message': f'Source {source} selected successfully'}), 200

		@app.route('/select_device', methods=['POST'])
		def select_device():
			data = request.get_json()
			device = data.get('device')

			if not device:
				return jsonify({'error': 'No device provided'}), 400

			self.init(self.model_path, self.input, device, self.data_type)

			return jsonify({'message': f'Device {device} selected successfully'}), 200

		@app.route('/select_model', methods=['POST'])
		def select_model():
			data = request.get_json()
			model = data.get('model')
			if not model:
				return jsonify({'error': 'No model provided'}), 400
			
			self.init(model, self.input, self.device, self.data_type)

			return jsonify({'message': f'Model {model} selected successfully'}), 200

		@app.route('/select_precision', methods=['POST'])
		def select_precision():
			data = request.get_json()
			data_type = data.get('precision')
			if not data_type:
				return jsonify({'error': 'No precision provided'}), 400
			
			self.init(self.model_path, self.input, self.device, data_type)
			
			return jsonify({'message': f'Precision {data_type} selected successfully'}), 200

		self.frames_number = 0
		self.start_time = perf_counter()
		self.running = True
		self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

	def video_stream(self):

		while self.cap is None:
			time.sleep(0.2)

		self.cv.acquire()
		while self.running:
			self.cv.release()
			frame = self.cap.read()
			self.cv.acquire()
			if self.model and frame is not None:
				self.model.predict(frame.copy())

			self.cv.release()
			while not self.queue.empty():
				frame = self.queue.get()
				if frame is not None:
					self.frames_number += 1
					ret, buffer = cv2.imencode('.jpg', frame)
					frame = buffer.tobytes()
					yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
					if not self.running:
						break
			self.cv.acquire()

		self.cap = None
		self.cv.notify_all()
		self.cv.release()

def main():
	app = ObjectDetector()
	app.run()

if __name__ == "__main__":
	fire.Fire(main)
