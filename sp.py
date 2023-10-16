from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import dlib 
import cv2
import time 
import numpy as np

im_path = '/home/jakhon37/myprojects/DECA/DECA_source/inputs/test_img/ac.jpg'
im_path = '/home/jakhon37/myprojects/DECA/DECA_source/inputs/test_img/idy/ad.jpg'
# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image = cv2.imread(im_path)
# image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# perform face detection using dlib's face detector
start = time.time()
print("[INFO[ performing face detection with dlib...")
bbox = detector(rgb, 1)
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))
print("[INFO] {} faces detected".format(len(bbox)))
print("[INFO] {} faces cords".format(bbox))
x,y,w,h = bbox[0].left(), bbox[0].top(), bbox[0].width(), bbox[0].height()
print(x,y,w,h)
# Process image
dataset = 'wflw'
processor = SPIGAFramework(ModelConfig(dataset))
features = processor.inference(image, [[x,y,w,h]])
print(features.keys())

import copy
from spiga.demo.visualize.plotter import Plotter

# Prepare variables
x0,y0,w,h = x,y,w,h
canvas = copy.deepcopy(image)
landmarks = np.array(features['landmarks'][0])
headpose = np.array(features['headpose'][0])

# Plot features
plotter = Plotter()
canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)

# Show image results
(h, w) = canvas.shape[:2]
canvas = cv2.resize(canvas, (512, int(h*512/w)))

cv2.imwrite('canvas.jpg', canvas)
# cv2.imshow('canvas', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow(canvas)
# cv2.imshow('canvas', canvas)
# cv2.waitKey(10)
# cv2.destroyAllWindows()

print('Done')