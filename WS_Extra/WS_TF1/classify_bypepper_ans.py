# --- coding: utf-8 ---
import argparse
import numpy as np
import pickle
from naoqi import ALProxy
import tensorflow as tf
import cv2

# Pepper の IP アドレスを引数で取得
parser = argparse.ArgumentParser()
parser.add_argument('ip_ad', help='my IP address(ifconfig)')
args = parser.parse_args()
ip_ad = args.ip_ad


# ラベルID(int) -> ラベル名(string)変換用クラス
class NodeLookup(object):
  def __init__(self):
    # node_lookup : dict (ID(int) -> label(string))
    self.node_lookup = {}
    label_lookup_path = 'inception-2015-12-05/imagenet2012_id_to_label.txt' # label file 読み込み
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)
    lines = tf.gfile.GFile(label_lookup_path).readlines()
    for line in lines:
      key, label = line.strip().split('\t')
      self.node_lookup[int(key)] = label
    print self.node_lookup

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


# ID -> label 辞書 Class の作成
node_lookup = NodeLookup()


# 保存された GraphDef から Tensorflow graph の作成
with tf.gfile.FastGFile('inception-2015-12-05/classify_image_graph_def.pb', 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')


# NAOqi module の取得
tts = ALProxy('ALTextToSpeech', ip_ad, 9559)
tts.setLanguage('English')
videoDevice = ALProxy('ALVideoDevice', ip_ad, 9559)

# 上部カメラの監視
AL_kTopCamera = 0
AL_kQVGA = 2   # 0: kQQVGA (160x120), 1: kQVGA (320x240), 2: kVGA (640x480)
AL_kBGRColorSpace = 13   # 0: kYuv, 9: kYUV422, 10: kYUV, 11: kRGB, 12: kHSY, 13: kBGR
fps = 5   # 5, 10, 15, 30
nameID = 'classify_TF'
videoDevice.unsubscribe(nameID) # 前回実行時のプロセスが残っていたら終了
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

    # 画像を一度JPEGで保存
    cv2.imwrite('tmp.jpg', image)

    # JPEGから読み出し
    image_data = tf.gfile.FastGFile('tmp.jpg', 'rb').read()

    # Tensorflow Session 開始
    with tf.Session() as sess:
      # 抽出層の指定
      # 'softmax:0': 1000 labels の正規化された prediction.
      # 'pool_3:0': (最終layer前のlayer) 2048次元の画像特徴量(float)
      softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
      predictions = sess.run(softmax_tensor,
                             {'DecodeJpeg/contents:0': image_data}) # 実行
      # 'DecodeJpeg/contents:0': 画像 (JPEGから読み込んだもの).
      summary_writer = tf.summary.FileWriter('./graphs', graph=sess.graph) # グラフ保存
      predictions = np.squeeze(predictions) # squeeze:大きさ1の次元を除去

      node_id = predictions.argsort()[-1] # 最も確率が高いラベル
      string = node_lookup.id_to_string(node_id)
      string = string.split(', ')[0] # ラベル名の「,」以降を除去
      score = predictions[node_id]
      print('%s (score = %.5f)' % (string, score))
      tts.say(string) # Pepperでラベル名を発声

    k = cv2.waitKey(50);
    if k == ord('q'):  break; #「q」を入力して終了

tts.setLanguage('Japanese')
videoDevice.unsubscribe(nameID)


"""
image[0] : [int] with of the image
image[1] : [int] height of the image
image[2] : [int] number of layers of the image
image[3] : [int] colorspace of the image
image[4] : [int] time stamp in second
image[5] : [int] time stamp in microsecond (and under second)
image[6] : [int] data of the image
image[7] : [int] camera ID
image[8] : [float] camera FOV left angle (radian)
image[9] : [float] camera FOV top angle (radian)
image[10]: [float] camera FOV right angle (radian)
image[11]: [float] camera FOV bottom angle (radian)
"""
