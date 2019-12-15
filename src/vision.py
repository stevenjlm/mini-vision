import webcolors
from sklearn.cluster import KMeans
from collections import Counter

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import yaml
import copy
import datetime

""" logging """
import logging

log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
log = logging.getLogger('bots')
log.setLevel(logging.DEBUG)

def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)

terminal_handler = logging.StreamHandler()
log_name = timeStamped('vision.log')
file_handler = logging.FileHandler('../log/' + log_name)
terminal_handler.setFormatter(log_format)
if (log.hasHandlers()):
    log.handlers.clear()
log.addHandler(terminal_handler)
log.addHandler(file_handler)

""" Vision Tools singleton class """

class VisionTools:

    class __VisionTools:
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier("../cascade/haarcascade_frontalface_default.xml")
            self.labelsPath = os.path.sep.join(["../yolo/", "coco.names"])
            self.LABELS = open(self.labelsPath).read().strip().split("\n")

            # derive the paths to the YOLO weights and model configuration
            self.weightsPath = os.path.sep.join(["../yolo/", "yolov3.weights"])
            self.configPath = os.path.sep.join(["../yolo/", "yolov3.cfg"])

            # load our YOLO object detector trained on COCO dataset (80 classes)
            log.info("Loading YOLO from disk...")
            self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

            # load comments from yaml
            self.stream = open('cv_comm.yaml', 'r')
            self.cv_comms = yaml.load(self.stream)

            # determine only the *output* layer names that we need from YOLO
            self.ln = self.net.getLayerNames()
            self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    instance = None
    def __init__(self):
        if not VisionTools.instance:
            VisionTools.instance = VisionTools.__VisionTools()

    def __getattr__(self, name):
        return getattr(self.instance, name)

""" Vision system """

class VisionSystem:
    def __init__(self, in_img, tools):
        self.img = in_img
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.tools = tools
        self.full_comment = ""
        self.pr_img = None

    def process_task(self, vision_task):
        vision_task.run(self)

    def dump_pr(self):
        image_file_name = timeStamped('processed.png')
        cv2.imwrite('../data/processed/' + image_file_name, self.pr_img, [cv2.IMWRITE_PNG_COMPRESSION, 8])

    def produce_comment(self):
        return self.full_comment

""" Vision task abstract class """

class VisionTask:
    def __init__(self):
        pass

""" Find the dominant color in the image """

class DominantColor(VisionTask):

    def run(system, k=5, image_processing_size = None):
        """
        takes an image as input
        returns the dominant color of the image as a list

        dominant color is found by running k means on the 
        pixels & returning the centroid of the largest cluster

        processing time is sped up by working with a smaller image; 
        this resizing can be done with the image_processing_size param 
        which takes a tuple of image dims as input

        >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
        [56.2423442, 34.0834233, 70.1234123]
        """
        #resize image if new dims provided
        if image_processing_size is not None:
            image = cv2.resize(system.img, image_processing_size, 
                                interpolation = cv2.INTER_AREA)

        #reshape the image to be a list of pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        #cluster and assign labels to the pixels 
        clt = KMeans(n_clusters = k)
        labels = clt.fit_predict(image)

        #count labels to find most popular
        label_counts = Counter(labels)

        #subset out most popular centroid
        i = 0
        log.info(label_counts)
        for i in range(4):
            dominant_color = clt.cluster_centers_[label_counts.most_common(4)[i][0]]
            color = get_colour_name(list(dominant_color)[::-1])
            if color != "gray" and color != "black" and color != "silver" and color!= "white":
                log.info("Color is %s" % color)
                system.full_comment += "There's a " + color + " tinge to the image. "
        return

def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.css21_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

""" Colorfulness """

class Colorfulness(VisionTask):

    def run(self, system):
        # Colorfullness
        colorfness = self.image_colorfulness(system.img)
        log.info("Colorfulness: %f" % colorfness)
        if colorfness > 52.0:
            system.full_comment += system.tools.cv_comms['colorful']
        if colorfness < 1.0:
            system.full_comment += system.tools.cv_comms['bw']

    def image_colorfulness(self, image):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype("float"))

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))

        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)

""" Contrast """
class ContrastTones(VisionTask):

    def run(self, system):
        mean_s = np.mean(system.hsv[:,:,2])
        std_s = np.std(system.hsv[:,:,2])
        log.info("Mean and std value %f, %f" % (mean_s, std_s))
        if std_s > 80.0:
            if system.full_comment == "":
                system.full_comment += system.tools.cv_comms['new-contrast']
            else:
                system.full_comment += system.tools.cv_comms['second-contrast']

        histr = cv2.calcHist([system.img],[0],None,[256],[0,256])
        sum_low = np.sum(histr[:65])
        sum_high = np.sum(histr[190:])
        system.peak = np.max(histr)
        bright_dark = 0
        if mean_s < 120 and sum_low > 2.7 * sum_high:
            system.full_comment += system.tools.cv_comms['dark']
            bright_dark = 1
        if mean_s > 120 and sum_high > 2.2 * sum_low:
            system.full_comment += system.tools.cv_comms['bright']
            bright_dark =1
        if np.abs(sum_high / sum_low) < .05:
            if bright_dark == 0:
                system.full_comment += system.tools.cv_comms['new-balance']
            else:
                system.full_comment += system.tools.cv_comms['old-balance']
            log.info("Sum low and high %f, %f" % (sum_low, sum_high))

""" Fade detection """
class Fade(VisionTask):

    def run(self, system):
        log.info("Testing for image fade")
        log.info(system.image_gray.shape)
        mn = np.amin(system.image_gray)
        comm = False
        if mn > 30.0:
            comm = True
            log.info("FADED")
            p = random.randint(0,3)
            if p <= 1:
                system.full_comment += system.tools.cv_comms['fade']
        return

""" Faces """
class Faces(VisionTask):

    def run(self, system):
        # detect all the faces in the image
        faces = system.tools.face_cascade.detectMultiScale(system.image_gray)
        log.info(faces)
        # print the number of faces detected
        log.info("%d faces detected in the image." % len(faces))
        system.pr_img = copy.deepcopy(system.img)
        for (x,y,w,h) in faces:
            cv2.rectangle(system.pr_img,(x,y),(x+w,y+h),(255,0,0),2)
        system.dump_pr()

        if len(faces) == 1:
            if "compisition" not in system.full_comment:
                self.face_composition(faces[0], system)
            if system.full_comment == "":
                system.full_comment += system.tools.cv_comms['one-face']
            else:
                system.full_comment += system.tools.cv_comms['old-one-face']
        elif len(faces) > 1:
            system.full_comment += system.tools.cv_comms['faces']

    def face_composition(self, position, system):
        x, y, w, h = position
        wid = system.img.shape[1]
        relx = (x + w/2.0) / wid
        log.info("Relative position")
        log.info(relx)
        if relx > .45 and relx < .55:
            system.full_comment += system.tools.cv_comms['comp-central']
            log.info("Central composition")
        elif relx > .28 and relx < .38:
            system.full_comment += system.tools.cv_comms['comp-thirds']
            log.info("Left third composition")
        elif relx > .61 and relx < .71:
            system.full_comment += system.tools.cv_comms['comp-thirds']
            log.info("Right third composition")


""" Object Recon """

class ObjRecon(VisionTask):

    def run(self, system):
        # Objects
        blob = cv2.dnn.blobFromImage(system.img, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        system.tools.net.setInput(blob)
        layerOutputs = system.tools.net.forward(system.tools.ln)

        # loop over each of the layer outputs
        label_count = [0 for lbl in system.tools.LABELS]
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                lbl = system.tools.LABELS[classID]
                if confidence > 0.0:
                    log.info("DETECTED: %s with conf score %f" % (lbl, confidence))
                if confidence > 0.85:
                    label_count[classID] += 1
        self.gen_comments(label_count, system)

    def gen_comments(self, label_count, system):
        idx = 0
        objects = 0
        for count in label_count:
            extention = ""
            if system.tools.LABELS[idx] == "person":
                if count == 1:
                    extention = system.tools.cv_comms['one-person']
                elif count > 3:
                    extention = system.tools.cv_comms['people']
            else:
                if count == 1:
                    if objects == 0:
                        extention = system.tools.cv_comms['new-obj']
                        extention += system.tools.LABELS[idx] + "! "
                        objects = 1
                    elif objects < 4:
                        extention = system.tools.cv_comms['more-obj']
                        extention += system.tools.LABELS[idx] + "! "
                elif count > 1:
                    if objects == 0:
                        extention = system.tools.cv_comms['new-objs']
                        extention += str(count) + " "
                        extention += system.tools.LABELS[idx] + "s! "
                        objects = 2
                    elif objects < 4:
                        extention = system.tools.cv_comms['more-objs']
                        extention += str(count) + " "
                        extention += system.tools.LABELS[idx] + "s! "
            system.full_comment += extention
            idx += 1
        return

""" Natural Image Test """
class NaturalImg(VisionTask):

    def run(self, system):
        histr = cv2.calcHist([system.img],[0],None,[256],[0,256])
        system.peak = np.max(histr)
        if system.peak > 32800:
            system.full_comment += system.tools.cv_comms['natural']

""" OpenCV metrics """
def opencv_metrics(img):
    tools = VisionTools()
    system = VisionSystem(img, tools)

    tasks = [Colorfulness(), ContrastTones(),
             Fade(), Faces(), ObjRecon()]
    for task in tasks:
        system.process_task(task)

    log.info("Len comment %d" % len(system.full_comment)) 
    return system.full_comment

if __name__ == '__main__':
    # reads an input image 
    img = cv2.imread('../img/img1.jpg')
    cv_comment = opencv_metrics(img)
    log.info("Final comment: %s" % cv_comment)
