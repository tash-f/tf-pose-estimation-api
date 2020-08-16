import argparse
import logging
import sys
import time
import falcon
import os
import matplotlib.pyplot as plt

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class OpenPoseInf(object):
    def __init__(self,loaded_model):
        self.model = loaded_model
    
    def on_post(self, req, resp):
        # TODO: refactor if possible to avoid writing/reading from disk
        img = 'tmp_img.jpg'
        with open(img, 'wb') as image_file: 
            while True:
                chunk = req.stream.read(4096)
                if not chunk:
                    break
                image_file.write(chunk)
        self.__inference('tmp_img.jpg')
        resp.stream = open("inf.png",'rb')
        resp.stream_len = os.path.getsize("inf.png")
        resp.content_type = falcon.MEDIA_PNG
        resp.status = falcon.HTTP_200

    def __inference(self,image_api):
        # estimate human poses from a single image !
        image = common.read_imgfile(image_api,None,None)
        if image is None:
            logger.error('Image can not be read, path=%s' % image_api)
            sys.exit(-1)

        t = time.time()
        #global e
        humans = self.model.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        elapsed = time.time() - t

        logger.info('inference image: %s in %.4f seconds.' % (image_api, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        try:
            # show network output
            bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            bgimg = cv2.resize(bgimg, (self.model.heatMat.shape[1], self.model.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

            # show network output
            plt.imsave('inf.png',bgimg)

        except Exception as e:
            logger.warning('return image error, %s' % e)



app = falcon.API()

model = 'cmu' # mobilenet_thin, mobilenet_v2_large, mobilenet_v2_small
resize = '0x0' # Recommends : 432x368 or 656x368 or 1312x736
resize_out_ratio = 4.0 # default 1.0?

w, h = model_wh(resize)

if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

openposeinf = OpenPoseInf(e)
app.add_route('/poseinference',openposeinf)

