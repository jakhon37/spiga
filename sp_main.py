
import dlib 
import cv2
import time 
import numpy as np
from imutils import face_utils
import copy
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from spiga.demo.visualize.plotter import Plotter
import os 
class SPI:
    
    def __init__(self, debug=False, compare = False):
        # if self.debug: print('|----- SPIGA init') 
        self.debug = debug
        deca_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.detector = dlib.get_frontal_face_detector()
        path_lnd = os.path.join(deca_root, 'data/other_weights/shape_predictor_68_face_landmarks.dat')
        if compare:
            self.predictor = dlib.shape_predictor(path_lnd) #model_cfg.dlib_model_path
        dataset = '300wpublic'#''#'wflw'300wpublic
        self.processor = SPIGAFramework(ModelConfig(dataset), debug=self.debug)
    
    def resize_im(self, img, size):
        max_dim = 512
        try:
            height, width = img.shape[:2]
            ratio = max_dim / height if height > width and height<512 else max_dim / width
            resized_img = cv2.resize(img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_AREA)
        except AttributeError:
            width, height = img.size
            ratio = max_dim / height if height > width else max_dim / width
            resized_img = img.resize((int(width * ratio), int(height * ratio)), Image.BILINEAR)
        except:
            if self.debug: print('|----- error in resize')
            resized_img = img
        return resized_img
    
    def rect_viz(self, im, points, copy=True):
        if copy:
            im = im.copy()        
        pnt1, pnt2 =  points[:2], (points[2] + points[0], points[3] + points[1])
        if self.debug: print("|----- Bounding box (x,y,x2,y2):", pnt1, pnt2)
        cv2.rectangle(im, pnt1, pnt2, (0, 255, 0), 1)
        # cv2.imshow('rect image', im)
        # cv2.waitKey(0)
        return im
        
    def point_viz(self, im, points, copy=True, viz = False):
        if copy:
            im = im.copy()    
        h, w = im.shape[:2]
        p_size = int(h / 300)    
        # if self.debug: print('|----- points: ', (points))
        for position, point  in enumerate(points):
            # print(point, position)
            xx, yy = point
            # print(f'{position} x: {xx} / y: {yy}')
            xx, yy = int(xx), int(yy)
            cv2.circle(im, (xx, yy), p_size, (0, 0, 255), -1)
        if viz:
            im = self.resize_im(im, 512)
            cv2.imshow('image', im)
            cv2.waitKey(0)
        return im   

    def stack_ims(self, img_list):
        imr_list = []
        for im in img_list:
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            r_im = self.resize_im(im, 512)
            imr_list.append(r_im)
        merged_image = cv2.hconcat(imr_list)
        cv2.imshow('Merged Image', merged_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_face(self, image):
        # if self.debug: print("|----- [INFO[ performing face detection with dlib...")
        bbox = self.detector(image, 1)
        # if self.debug: print("|----- [INFO] {} faces detected".format(len(bbox)))
        # if self.debug: print("|----- [INFO] {} faces cords".format(bbox))
        return bbox
    
    def dlib_landmark(self, image, bbox):
        shape = self.predictor(image, bbox[0])
        shape = face_utils.shape_to_np(shape)
        # if self.debug: print('|----- landmarks dlib: ', len(shape))
        return shape
    
    def plot_spiga(self, canvas, x_y_w_h, features, copy=True):
        if copy:
            canvas = canvas.copy()        
        # Prepare variables
        x0,y0,w,h = x_y_w_h
        landmarks = np.array(features['landmarks'][0])
        headpose = np.array(features['headpose'][0])
        plotter = Plotter()
        canvas = plotter.landmarks.draw_landmarks(canvas, landmarks)
        canvas = plotter.hpose.draw_headpose(canvas, [x0,y0,x0+w,y0+h], headpose[:3], headpose[3:], euler=True)
        canvas = canvas[0]
        (h, w) = canvas.shape[:2]
        # canvas = cv2.resize(canvas, (512, int(h*512/w)))

        cv2.imwrite('canvas.jpg', canvas)
        if self.debug: print('|----- canvas shape: ', canvas.shape)
        # cv2.imshow('canvas', canvas)  
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return canvas
        
            
    def inference(self, image):
        # image = cv2.imread(im_path)
        # image = imutils.resize(image, width=600)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        bbox = self.detect_face(rgb)
        landmark = self.dlib_landmark(rgb, bbox)
        dlib_lnd = self.point_viz(image, landmark)
        end = time.time()  
        if self.debug: print("|----- [INFO] dlib face detection took {:.4f} seconds".format(end - start))
        x_y_w_h = int(bbox[0].left()), int(bbox[0].top()), int(bbox[0].width()), int(bbox[0].height())
        dlib_lnd_box = self.rect_viz(dlib_lnd, x_y_w_h, copy=True)
        
        if self.debug: print('|----- start SPIGA')
        features = self.processor.inference(image, [x_y_w_h])
        if self.debug: print('|----- features: ', len(features))
        sg_plt_lnd = self.plot_spiga(image, x_y_w_h, features)
        landmarks = np.array(features['landmarks'][0])
        if self.debug: print('|----- landmarks spiga: ', landmarks.shape)
        sg_cv_lnd = self.point_viz(image, landmarks)
        img_list= [dlib_lnd, sg_cv_lnd,  dlib_lnd_box, sg_plt_lnd]
        self.stack_ims(img_list)
        return landmarks



    def spiga_inference(self, image, bbox=None, vis=False):

        # image = cv2.imread(im_path)
        # image = imutils.resize(image, width=600)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        if not bbox: bbox = self.detect_face(rgb)
        if self.debug: print('|----- bbox: ', bbox)
        if len(bbox) != 0:
            x_y_w_h = int(bbox[0].left()), int(bbox[0].top()), int(bbox[0].width()), int(bbox[0].height())
        else:
            if self.debug: print('|----- no face detected') 
            x_y_w_h = 10, 10, 20, 20
        if self.debug: print('|----- start SPIGA')
        features = self.processor.inference(image, [x_y_w_h])
        if self.debug: print('|----- features: ', len(features))
        if vis: self.plot_spiga(image, x_y_w_h, features)
        
        landmarks = np.array(features['landmarks'][0])
        if self.debug: print('|----- landmarks spiga: ', landmarks.shape)
        if vis: self.point_viz(image, landmarks, viz=vis)
        
        # img_list= [dlib_lnd, sg_cv_lnd,  dlib_lnd_box, sg_plt_lnd]
        # self.stack_ims(img_list)
        
        end = time.time()  
        return landmarks, bbox

    def spiga_headPose(self, image, bbox, vis=False):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        # bbox = self.detect_face(rgb)
        if self.debug: print('|----- bbox: ', bbox)
        if len(bbox) != 0:
            x_y_w_h = int(bbox[0].left()), int(bbox[0].top()), int(bbox[0].width()), int(bbox[0].height())
        else:
            if self.debug: print('|----- no face detected') 
            x_y_w_h = 10, 10, 20, 20
        if self.debug: print('|----- start SPIGA')
        features = self.processor.inference(image, [x_y_w_h])
        if self.debug: print('|----- features: ', len(features))
        if vis: self.plot_spiga(image, x_y_w_h, features)
        headpose = np.array(features['headpose'][0])
        landmarks = np.array(features['landmarks'][0])
        if self.debug: print('|----- landmarks spiga: ', landmarks.shape)
        if vis: self.point_viz(image, landmarks, viz=vis)
        end = time.time()  
        self.debug:  print("|----- [INFO] dlib face detection took {:.4f} seconds".format(end - start))
        return landmarks,headpose
    


    def s_inference(self, image):

        # image = cv2.imread(im_path)
        # image = imutils.resize(image, width=600)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = self.detect_face(rgb)
        x_y_w_h = int(bbox[0].left()), int(bbox[0].top()), int(bbox[0].width()), int(bbox[0].height())
        features = self.processor.inference(image, [x_y_w_h])
        # if self.debug: print('|----- features: ', len(features))
        # self.plot_spiga(image, x_y_w_h, features)
        landmarks = np.array(features['landmarks'][0])
        # if self.debug: print('|----- landmarks spiga: ', landmarks.shape)
        # self.point_viz(image, landmarks)
        return landmarks
        
        
if __name__ == '__main__':
    sp = SPI(debug=True, compare = True)
    root = os.path.dirname(os.path.abspath(__file__))
    im_path = os.path.join(root, 'assets/test.jpg')
    debug = True
    # sp.spiga_inference(im_path)
    # sp.inference(im_path)
    
    image = cv2.imread(im_path)
    # landmarks = sp.s_inference(image)
    landmarks = sp.inference(image)
    if debug: print('|----- landmarks: ', landmarks.shape)