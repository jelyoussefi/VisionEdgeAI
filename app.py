import os, sys, time, signal, platform, psutil, subprocess, logging
import fire
import cv2
import numpy as np
import csv
import io
from threading import Condition
from queue import Queue, Empty, Full
from time import perf_counter
from flask import Flask, render_template, jsonify, request, Response
from flask_bootstrap import Bootstrap
from utils.images_capture import VideoCapture

from utils.yolov8_model import YoloV8Model
from utils.ssd_model import SSDModel
import threading

# Disable Flask's default request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set level to ERROR to hide access logs

models = {
	"yolov8n": {
		"model": "yolov8n",
		"adapter": YoloV8Model
	},
	"yolov8s": {
		"model": "yolov8s",
		"adapter": YoloV8Model
	},
	"yolov8m": {
		"model": "yolov8m",
		"adapter": YoloV8Model
	},
	"person-detection": {
		"model": "pedestrian-detection-adas-0002",
		"adapter": SSDModel
	}
}

class ObjectDetector:
	def __init__(self):
		self.app = Flask(__name__)
		self.port = 5000
		self.running = True
		self.cv = Condition()
		self.queue = Queue(maxsize=16)  # Set the queue size limit
		self.upload_folder = '/workspace/videos'
		os.makedirs(self.upload_folder, exist_ok=True)
		self.model = self.model_name = self.device = self.input = self.data_type = self.cap = None

	def init(self, model_name, input, device="GPU", data_type="FP16"):

		with self.cv:
			if (model_name != self.model_name) or (device != self.device) or (data_type != self.data_type):
				self.model_name = model_name
				self.device = device
				self.data_type = data_type
				if model_name and device and data_type:
					model_path = f'/opt/models/{models[model_name]["model"]}/{data_type}/{models[model_name]["model"]}.xml'
					adapter = models[model_name]['adapter']
					if self.model:
						self.model.shutdown()  # Safely shut down existing model
					try:
						self.model = adapter(model_path, device, data_type, self.callback_function)
					except Exception as e:
						print(f"Cannot init the model: {e}")

					self.queue.queue.clear()

			if input != self.input:
				self.input = input
				self.queue.queue.clear()
				self.cap = VideoCapture(input)
			
			self.cv.notify_all()  # Notify other threads that initialization is complete

	def callback_function(self, frame):
		if self.running:
			try:
				self.queue.put(frame, timeout=0.1)  # Try to add frame with timeout to avoid blocking
			except Full:
				print("Queue is full, dropping frame.")

	def video_stream(self):
		while self.running:
			if self.cap is not None:
				# Capture a frame from the video input
				frame = self.cap.read()
				if frame is not None and self.model is not None:
					# Perform model prediction on the frame, triggering the callback
					self.model.predict(frame)

			# Try to get a frame from the queue with a short timeout
			try:
				frame = self.queue.get(timeout=0.01)
			except Empty:
				frame = None

			if frame is not None:
				ret, buffer = cv2.imencode('.jpg', frame)
				frame = buffer.tobytes()
				yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

			time.sleep(0.02)

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
			model_names = list(models.keys())
			default_device = "GPU"
			default_precision = "FP16"
			default_model = model_names[0] if model_names else "No models available"
			default_source = os.path.join(self.upload_folder, default_file)

			self.init(default_model, default_source, default_device, default_precision)

			return render_template('index.html', 
									default_device=default_device, 
									default_model=default_model,
									default_precision=default_precision, 
									default_file=default_file,
									cpu_model=cpu_model, 
									gpu_model=gpu_model,
									model_names=model_names  
									)

		@app.route('/video_feed')
		def video_feed():
			return Response(self.video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

		@app.route('/metrics', methods=['GET'])
		def get_metrics():
			try:
				cpu_percent = 0 #psutil.cpu_percent(interval=None)
				power_data = 0 #self.get_power_consumption()
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

			self.init(self.model_name, input, self.device, self.data_type)

			return jsonify({'message': f'Source {source} selected successfully'}), 200

		@app.route('/select_device', methods=['POST'])
		def select_device():
			data = request.get_json()
			device = data.get('device')

			if not device:
				return jsonify({'error': 'No device provided'}), 400

			self.init(self.model_name, self.input, device, self.data_type)

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
			
			self.init(self.model_name, self.input, self.device, data_type)
			
			return jsonify({'message': f'Precision {data_type} selected successfully'}), 200

		self.frames_number = 0
		self.start_time = perf_counter()
		self.running = True
		self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

		@app.route('/shutdown', methods=['POST'])
		def shutdown_server():
			if request.remote_addr != '127.0.0.1':  # Optional: restrict shutdown to local requests only
				return jsonify({"error": "Unauthorized"}), 403
			
			func = request.environ.get('werkzeug.server.shutdown')
			if func is None:
				raise RuntimeError("Not running with the Werkzeug Server")
			func()
			return jsonify({"message": "Server shutting down..."})

	def shutdown(self, signum=None, frame=None):
		print("Shutting down gracefully...")
		self.running = False  # Set the running flag to False to stop threads

		# Shut down the Flask server
		try:
			# Use requests to call the shutdown route from within the application
			requests.post("http://127.0.0.1:5000/shutdown")
		except Exception as e:
			print("Error shutting down Flask server:", e)

		# If there’s a model running, shut it down
		if self.model:
			self.model.shutdown()
		print("Server has been stopped.")

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

def main():
	app = ObjectDetector()
	app.run()

if __name__ == "__main__":
	fire.Fire(main)
