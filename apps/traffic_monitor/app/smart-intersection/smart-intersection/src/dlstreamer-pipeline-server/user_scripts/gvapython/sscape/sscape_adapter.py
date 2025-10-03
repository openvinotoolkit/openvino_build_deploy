# Copyright (C) 2024 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you ("License"). Unless the License provides otherwise,
# you may not use, modify, copy, publish, distribute, disclose or transmit
# this software or the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the License.

import base64
import json
import logging
import math
import os
import struct
import time
from collections import defaultdict
from datetime import datetime
from uuid import getnode as get_mac

import cv2
import ntplib
import numpy as np
import paho.mqtt.client as mqtt
from pytz import timezone

from utils import publisher_utils as utils

ROOT_CA = os.environ.get('ROOT_CA', '/run/secrets/certs/scenescape-ca.pem')
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
TIMEZONE = "UTC"

def getMACAddress():
  if 'MACADDR' in os.environ:
    return os.environ['MACADDR']

  a = get_mac()
  h = iter(hex(a)[2:].zfill(12))
  return ":".join(i + next(h) for i in h)

class PostDecodeTimestampCapture:
  def __init__(self, ntpServer=None):
    self.log = logging.getLogger('SSCAPE_ADAPTER')
    self.log.setLevel(logging.INFO)
    self.ntpClient = ntplib.NTPClient()
    self.ntpServer = ntpServer
    self.lastTimeSync = None
    self.timeOffset = 0
    self.ts = None
    self.timestamp_for_next_block = None
    self.fps = 5.0
    self.fps_alpha = 0.75 # for weighted average
    self.last_calculated_fps_ts = None
    self.fps_calc_interval = 1 # calculate fps every 1s
    self.frame_cnt = 0

  def processFrame(self, frame):
    now = time.time()
    self.frame_cnt += 1
    if not self.last_calculated_fps_ts:
      self.last_calculated_fps_ts = now
    if (now - self.last_calculated_fps_ts) > self.fps_calc_interval:
      self.fps = self.fps * self.fps_alpha + (1 - self.fps_alpha) * (self.frame_cnt / (now - self.last_calculated_fps_ts))
      self.last_calculated_fps_ts = now
      self.frame_cnt = 0

    if self.ntpServer:
      # if ntpServer is available, check if it is time to recalibrate
      if not self.lastTimeSync or now - self.lastTimeSync > 1000 :
        response = self.ntpClient.request(host=self.ntpServer, port=123)
        self.timeOffset = response.offset
        self.lastTimeSync = now

    now += self.timeOffset
    self.timestamp_for_next_block = now
    frame.add_message(json.dumps({
      'postdecode_timestamp': f"{datetime.fromtimestamp(now, tz=timezone(TIMEZONE)).strftime(DATETIME_FORMAT)[:-3]}Z",
      'timestamp_for_next_block': now,
      'fps': self.fps
    }))
    return True

def computeObjBoundingBoxParams(pobj, fw, fh, x, y, w, h, xminnorm=None, yminnorm=None, xmaxnorm=None, ymaxnorm=None):
  # use normalized bounding box for calculating center of mass
  xmax, xmin = int(xmaxnorm * fw), int(xminnorm * fw)
  ymax, ymin = int(ymaxnorm * fh), int(yminnorm * fh)
  comw, comh = (xmax - xmin) / 3, (ymax - ymin) / 4

  pobj.update({
    'center_of_mass': {'x': int(xmin + comw), 'y': int(ymin + comh), 'width': comw, 'height': comh},
    'bounding_box_px': {'x': x, 'y': y, 'width': w, 'height': h}
  })

  return

def detectionPolicy(pobj, item, fw, fh):
  pobj.update({
    'category': item['detection']['label'],
    'confidence': item['detection']['confidence']
  })
  computeObjBoundingBoxParams(pobj, fw, fh, item['x'], item['y'], item['w'],item['h'],
                              item['detection']['bounding_box']['x_min'],
                              item['detection']['bounding_box']['y_min'],
                              item['detection']['bounding_box']['x_max'],
                              item['detection']['bounding_box']['y_max'])

  return

def reidPolicy(pobj, item, fw, fh):
  detectionPolicy(pobj, item, fw, fh)
  reid_vector = item['tensors'][1]['data']
  # following code snippet is from percebro/modelchain.py
  v = struct.pack("256f",*reid_vector)
  pobj['reid'] = base64.b64encode(v).decode('utf-8')
  return

def classificationPolicy(pobj, item, fw, fh):
  detectionPolicy(pobj, item, fw, fh)
  # todo: add configurable parameters(set tensor name)
  pobj['category'] = item['classification_layer_name:efficientnet-b0/model/head/dense/BiasAdd:0']['label']
  return

metadatapolicies = {
"detectionPolicy": detectionPolicy,
"reidPolicy": reidPolicy,
"classificationPolicy": classificationPolicy
}

class PostInferenceDataPublish:
  def __init__(self, cameraid, metadatagenpolicy='detectionPolicy', publish_image=False):
    self.cameraid = cameraid

    self.is_publish_image = publish_image
    self.is_publish_calibration_image = False
    self.setupMQTT()
    self.metadatagenpolicy = metadatapolicies[metadatagenpolicy]
    self.frame_level_data = {'id': cameraid, 'debug_mac': getMACAddress()}
    return

  def on_connect(self, client, userdata, flags, rc):
    if rc == 0:
      print(f"Connected to MQTT Broker {self.broker}")
      self.client.subscribe(f"scenescape/cmd/camera/{self.cameraid}")
      print(f"Subscribed to topic: scenescape/cmd/camera/{self.cameraid}")
    else:
      print(f"Failed to connect, return code {rc}")
    return

  def setupMQTT(self):
    self.client = mqtt.Client()
    self.client.on_connect = self.on_connect
    self.broker = "broker.scenescape.intel.com"
    self.client.connect(self.broker, 1883, 120)
    self.client.on_message = self.handleCameraMessage
    if ROOT_CA and os.path.exists(ROOT_CA):
      self.client.tls_set(ca_certs=ROOT_CA)
    self.client.loop_start()
    return

  def handleCameraMessage(self, client, userdata, message):
    msg = str(message.payload.decode("utf-8"))
    if msg == "getimage":
      self.is_publish_image = True
    elif msg == "getcalibrationimage":
      self.is_publish_calibration_image = True
    return

  def annotateObjects(self, img):
    objColors = ((0, 0, 255), (255, 128, 128), (207, 83, 294), (31, 156, 238))
    for otype, objects in self.frame_level_data['objects'].items():
      if otype == "person":
        cindex = 0
        # annotation of pose not supported
        #self.annotateHPE(frame, obj)
      elif otype == "vehicle" or otype == "bicycle":
        cindex = 1
      else:
        cindex = 2
      for obj in objects:
        topleft_cv = (int(obj['bounding_box_px']['x']), int(obj['bounding_box_px']['y']))
        bottomright_cv = (int(obj['bounding_box_px']['x'] + obj['bounding_box_px']['width']),
                        int(obj['bounding_box_px']['y'] + obj['bounding_box_px']['height']))
        cv2.rectangle(img, topleft_cv, bottomright_cv, objColors[cindex], 4)
    return

  def annotateFPS(self, img, fpsval):
    # code snippet is taken from annotateFPS method in percebro/videoframe.py
    fpsStr = f'FPS {fpsval:.1f}'
    scale = int((img.shape[0] + 479) / 480)
    cv2.putText(img, fpsStr, (0, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX,
            1 * scale, (0,0,0), 5 * scale)
    cv2.putText(img, fpsStr, (0, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX,
            1 * scale, (255,255,255), 2 * scale)
    return

  def buildImgData(self, imgdatadict, gvaframe, annotate):
    imgdatadict.update({
      'timestamp': self.frame_level_data['timestamp'],
      'id': self.cameraid
    })
    with gvaframe.data() as image:
      if annotate:
        self.annotateObjects(image)
        self.annotateFPS(image, self.frame_level_data['rate'])
      _, jpeg = cv2.imencode(".jpg", image)
    jpeg = base64.b64encode(jpeg).decode('utf-8')
    imgdatadict['image'] = jpeg

    return

  def buildObjData(self, gvadata):
    now = time.time()
    self.frame_level_data.update({
      'timestamp': gvadata['postdecode_timestamp'],
      'debug_timestamp_end': f"{datetime.fromtimestamp(now, tz=timezone(TIMEZONE)).strftime(DATETIME_FORMAT)[:-3]}Z",
      'debug_processing_time': now - float(gvadata['timestamp_for_next_block']),
      'rate': float(gvadata['fps'])
    })
    objects = defaultdict(list)
    if 'objects' in gvadata and len(gvadata['objects']) > 0:
      framewidth, frameheight = gvadata['resolution']['width'], gvadata['resolution']['height']
      for det in gvadata['objects']:
        vaobj = {}
        self.metadatagenpolicy(vaobj, det, framewidth, frameheight)
        otype = vaobj['category']
        vaobj['id'] = len(objects[otype]) + 1
        objects[otype].append(vaobj)
    self.frame_level_data['objects'] = objects

  def processFrame(self, frame):
    if self.client.is_connected():
      gvametadata, imgdatadict = {}, {}

      utils.get_gva_meta_messages(frame, gvametadata)
      gvametadata['gva_meta'] = utils.get_gva_meta_regions(frame)

      self.buildObjData(gvametadata)

      if self.is_publish_image:
        self.buildImgData(imgdatadict, frame, True)
        self.client.publish(f"scenescape/image/camera/{self.cameraid}", json.dumps(imgdatadict))
        self.is_publish_image = False

      if self.is_publish_calibration_image:
        if not imgdatadict:
          self.buildImgData(imgdatadict, frame, False)
        self.client.publish(f"scenescape/image/calibration/camera/{self.cameraid}", json.dumps(imgdatadict))
        self.is_publish_calibration_image = False

      self.client.publish(f"scenescape/data/camera/{self.cameraid}", json.dumps(self.frame_level_data))
      frame.add_message(json.dumps(self.frame_level_data))
    return True
