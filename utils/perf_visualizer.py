import os
import cv2
import numpy as np


def draw_perf(image:np.ndarray, device, fps, cpu_load):
	frame_size = image.shape[:-1]
	fontFace =  cv2.FONT_HERSHEY_SIMPLEX
	fontScale = 0.4
	thickness = 1
	margin = 15
	bcolor = (0,255,0)
	fcolor = (0,0,255)
	
	def circle(text, radius, pos=None, left=True, bcolor = (0,255,0), fcolor = (0,0,255), legend=""):
		textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]

		if pos is None:
			x = frame_size[1]/2
		elif left:
			x = margin + 2*(radius+5)*pos + radius/2
		else :
			x = frame_size[1] - margin - 2*(radius+5)*pos - radius/2

		center = (int(x), int(margin + radius / 2))
		cv2.circle(image, center, radius, bcolor, 1, cv2.LINE_AA)
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, text, textPos, fontFace, fontScale, fcolor, thickness, cv2.LINE_AA)

		textsize = cv2.getTextSize(legend, fontFace, fontScale, thickness)[0]
		center = (int(x), int(margin + radius*2))
		textPos = (int(center[0] - textsize[0]/2), int(center[1] + textsize[1]/2))
		cv2.putText(image, legend, textPos, fontFace, 0.4, (255,255,255), thickness, cv2.LINE_AA)

	# device name
	circle(device, 20)

	# fps
	fps = f"{int(fps)}"
	circle(fps, 20, 0, legend="fps")

	#cpu load
	if cpu_load is not None:
		cpu_load = f"{int(cpu_load)}"
		circle(cpu_load, 20, 0, False, legend="%cpu", fcolor=(255,0,0))

	
	

