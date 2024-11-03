import os, platform, queue, psutil, subprocess, random, logging
import fire
import cv2
import numpy as np
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
	def __init__(self, model, input, device="GPU", data_type="FP16"):
		self.app = Flask(__name__)
		self.port = 5000
		self.running = False
		self.cv = Condition()
		self.queue = queue.Queue(maxsize=0)  
		self.cpu_loads = deque(maxlen=120)
		self.upload_folder = '/workspace/videos'
		os.makedirs(self.upload_folder, exist_ok=True)
		self.init(model, input, device, data_type)

	def init(self, model, input, device="GPU", data_type="FP16"):
		self.cv.acquire()
		self.input = input
		self.model = YoloV8Model(model, device, data_type, self.callback_function)
		self.frame = None
		self.cap = VideoCapture(input)
		self.frames_number = 0
		self.cpu_loads.append(psutil.cpu_percent(0.1))
		self.start_time = perf_counter()
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

			return render_template('index.html', default_device=self.model.device, default_model=self.model.name,
								   default_precision=self.model.data_type, default_file=default_file,
								   cpu_model=cpu_model, gpu_model=gpu_model)

		@app.route('/video_feed')
		def video_feed():
			return Response(self.video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

		@app.route('/metrics', methods=['GET'])
		def get_metrics():
			try:
				cpu_percent = psutil.cpu_percent(interval=None)
				power_data = random.randint(15, 45)
				fps =  int(self.fps())
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


		@app.route('/metrics2', methods=['GET'])
		def get_metrics2():

			def extract_value_from_output(output, label):
				# Implement parsing logic based on pcm-power output format
				for line in output.splitlines():
					if label in line:
						return float(line.split()[-2])  # Adjust index as per output format
				return None
	
			try:
				# Get CPU load using psutil
				cpu_percent = psutil.cpu_percent(interval=1)

				# Run pcm-power command and capture the power data
				#result = subprocess.run(['pcm-power'], capture_output=True, text=True)
				#output = result.stdout

				# Extract power data (adjust based on actual pcm-power output format)
				#power_data = {
				#	'cpu_power': extract_value_from_output(output, 'CPU Power'),
				#	'dram_power': extract_value_from_output(output, 'DRAM Power')
				#}
				power_data = random.randint(15, 45)
				# Combine CPU and power data into one response
				metrics = {
					'cpu_percent': cpu_percent,
					'power_data': power_data
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

			print(f"Selected device: {device}")
			self.init(self.model.model_path, self.input, device, self.model.data_type)
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
			data_type = data.get('precision')
			if not data_type:
				return jsonify({'error': 'No precision provided'}), 400
			if data_type != self.model.data_type:
				print(f"Selected precision: {data_type}")
				self.init(self.model.model_path, self.input, self.model.device, data_type)
			return jsonify({'message': f'Precision {data_type} selected successfully'}), 200

		self.frames_number = 0
		self.start_time = perf_counter()
		self.running = True
		self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

	def fps(self):
		return self.frames_number/(perf_counter() - self.start_time)

	def video_stream(self):
		self.cv.acquire()
		while self.running:
			self.cv.release()
			frame = self.cap.read()
			self.cv.acquire()
			if frame is not None:
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

def main(model, input, device, **kwargs):
	app = ObjectDetector(model, input, device)
	app.run()

if __name__ == "__main__":
	fire.Fire(main)
