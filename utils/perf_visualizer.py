import time
import numpy as np
from threading import Thread, Condition
from collections import deque
import panel as pn
import psutil
from selenium import webdriver
from PIL import Image
import cv2
import io

# Initialize Panel components
pn.extension()

class PerfVisualizer():
    def __init__(self):
        self.running = False 
        self.cv = Condition()
        self.cpu_loads = deque(maxlen=120)

        self.cpu_gauge = pn.indicators.Gauge(
                                            name="CPU Load",
                                            value=0,
                                            bounds=(0, 100),
                                            format='{value}%',  # Show percentage format for CPU load
                                            colors=[(0.5, "green"), (0.8, "gold"), (1, "red")]  # Define colors as fractions of the range
                                        )
        self.html_file = "/tmp/panel_dashboard.html"

    def start(self):
        self.cv.acquire()
        self.running = True 
        self.proc = Thread(target=self.cpu_load_handler)
        self.proc.daemon = True
        self.proc.start()
        self.cv.release()


    def cpu_load(self):
        return np.average(self.cpu_loads);

    def cpu_load_handler(self):

        self.cpu_loads.append(psutil.cpu_percent(0.1))
        
        while self.running:
            self.cv.acquire()
            self.cpu_loads.append(psutil.cpu_percent(0))
            self.cv.release()
            time.sleep(0.5)