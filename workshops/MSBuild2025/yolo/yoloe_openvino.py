from ultralytics import YOLOE
from ultralytics import YOLO
import cv2

model_name="yoloe-11l-seg.pt"
ov_model_name="yoloe-11l-seg_openvino_model"

# Initialize a YOLOE model
model = YOLOE(model_name) 

# Set text prompt
names = ["person", "cup", "sunglasses", "black keyboard", "white keyboard"]
model.set_classes(names, model.get_text_pe(names))

# Dynamic shape is disabled for NPU. 
#Please enable dynamic shape if we are using CPU or GPU
model.export(format="openvino", dynamic=False, half=True)

model_ov = YOLO(ov_model_name)
video_cap = cv2.VideoCapture(0)
#video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
	ret, frame = video_cap.read()
	#can choose between intel:cpu, intel:gpu, or intel:npu
	results = model_ov.predict(frame,conf=0.25, device="intel:gpu")
	# Show results
	frame_out=results[0].plot()
	if not ret:
		break
	cv2.imshow("OpenVINO x YOLO-E Real-Time Seeing Anything", frame_out)
	if cv2.waitKey(1) == ord("q"):
		break

video_cap.release()
cv2.destroyAllWindows()
