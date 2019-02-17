# --- coding: utf-8 ---
import sys
sys.path.append("ssd_keras")

import argparse
import numpy as np 
import cv2

import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image as imgprocess
from keras.utils.visualize_util import plot

from ssd import SSD300
from ssd_utils import BBoxUtility

from naoqi import ALProxy

# Pepper の IP アドレスを引数で取得
parser = argparse.ArgumentParser()
parser.add_argument('ip_ad', help='my IP address(ifconfig)')
args = parser.parse_args()
ip_ad = args.ip_ad


# 各種設定
input_shape = (300, 300, 3)
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", 
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", 
               "sheep", "sofa", "train", "tvmonitor"]; # SSD_kerasのクラスラベル名
num_classes = len(class_names)

# 学習モデル読み込み
model = SSD300(input_shape, num_classes=num_classes)
model.load_weights('ssd_keras/weights_SSD300.hdf5')
bbox_util = BBoxUtility(num_classes)
plot(model, to_file='graphs/SSDmodel.png') # モデル描画

# バウンディングボックスの色設定
class_colors = []
for i in range(0, num_classes):
    # This can probably be written in a more elegant manner
    hue = 255*i/num_classes
    col = np.zeros((1,1,3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128 # Saturation
    col[0][0][2] = 255 # Value
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)


# 出力をバウンディングボックスに変換して画像に描画する関数
def draw_bbox_from_results(image, results):
    if len(results) > 0 and len(results[0]) > 0:
        det_label = results[0][:, 0]
        det_conf = results[0][:, 1]
        det_xmin = results[0][:, 2]
        det_ymin = results[0][:, 3]
        det_xmax = results[0][:, 4]
        det_ymax = results[0][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6] #0.6以上のものだけ取得
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            class_num = int(top_label_indices[i])
            class_name = class_names[class_num]

            # 画像に BOX を描画
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), 
                          class_colors[class_num], 2)
            text = class_name + " " + ('%.2f' % score)

            text_top = (xmin, ymin-10)
            text_bot = (xmin + 80, ymin + 5)
            text_pos = (xmin + 5, ymin)
            cv2.rectangle(image, text_top, text_bot, class_colors[class_num], -1)
            cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            print text

        return image



# NAOqi module の取得
videoDevice = ALProxy('ALVideoDevice', ip_ad, 9559)
videoDevice.unsubscribe('test')

# 上部カメラの監視
AL_kTopCamera = 0
AL_kQVGA = 2   # 0: kQQVGA (160x120), 1: kQVGA (320x240), 2: kVGA (640x480)
AL_kBGRColorSpace = 13   # 0: kYuv, 9: kYUV422, 10: kYUV, 11: kRGB, 12: kHSY, 13: kBGR
fps = 5   # 5, 10, 15, 30
nameID = 'test'
captureDevice = videoDevice.subscribeCamera(
    nameID, AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, fps)

width = 640
height = 480

 
while True:
  result = videoDevice.getImageRemote(captureDevice)  # Pepperから画像取得
  image = np.zeros((height, width, 3), np.uint8)

  if result == None:
    print 'cannot capture.'
  elif result[6] == None:
    print 'no image data string.'
  else:
    values = map(ord, list(result[6]))
    # Pepperから得られる画像情報は一列なのでxy,RGBにマッピング
    i = 0
    for y in range(0, height):
      for x in range(0, width):
        image.itemset((y, x, 0), values[i + 0])
        image.itemset((y, x, 1), values[i + 1])
        image.itemset((y, x, 2), values[i + 2])
        i += 3

    image = cv2.resize(image, (300,300))
    #cv2.imwrite("input.jpg",frame)

    # kerasに与えるための画像の前処理
    input_image = [imgprocess.img_to_array(image)]
    input_image = preprocess_input(np.array(input_image))

    prediction = model.predict(input_image)  # 実行
    results = bbox_util.detection_out(prediction)  # BBOXとして出力を処理
    result_image = draw_bbox_from_results(image, results) # BBOXを画像に描画

    cv2.imshow("pepper-camera-ssd", image) # 画像表示
    
    k = cv2.waitKey(5);
    if k == ord('q'):  break;
 
videoDevice.unsubscribe(nameID)
