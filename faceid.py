# Improt kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
class CamApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    # Run continously to get webcam feed
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250,:]

        # Flip horizantall and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture


    def preprocess(self,file_path):
        #read img from file path
        byte_img = tf.io.read_file(file_path)
        #load in the img
        img = tf.io.decode_jpeg(byte_img)
        #preprocess steps resizing the img
        img = tf.image.resize(img, (100,100))
        #scale img to be between 0 and 1
        img = img /255.0
        #return img
        return img
    
    # verification function to verify person
    def verify(self, *args):
        detection_threshold = 0.0
        verification_threshold =0.0

        #capture input img from webcom
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250,:]
        cv2.imwrite(SAVE_PATH, frame)

    #built result array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # make prediction
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # detection threshold metric above which a prediction is considered positive
        detection = np.sum(np.array(result) > detection_threshold)
        

        # verification threshold: proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # set verification text
        self.verification_label.text = 'verified' if verified ==True else 'unverified'

        # log out details
        Logger.info(result)
        Logger.info(np.sum(np.array(result) >0.2))
        Logger.info(np.sum(np.array(result) >0.3))
        Logger.info(np.sum(np.array(result) >0.4))
        Logger.info(np.sum(np.array(result) >0.5))

        return result,verified
    

if __name__ == "__main__":
    CamApp().run()