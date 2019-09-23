import base64

from flask import Flask, request, jsonify
import cv2
import tensorflow as tf
import numpy as np
from keras import models

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)


def preprocess_input(x):
    x = x.astype('float32')
    # 我是用cv2来读取的图片，其已经是BGR格式了
    mean = [103.939, 116.779, 123.68]
    x -= mean
    return x


def _get_imgs(img):
    interval = 5
    length = 67
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]


def get_text(img, offset=0):
    text = img[3:22, 120 + offset:177 + offset]
    text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    text = text / 255.0
    h, w = text.shape
    text.shape = (1, h, w, 1)
    return text


# 初始化模型
word_model = models.load_model('model.v2.0.h5', compile=False)
image_model = models.load_model('12306.image.model.h5', compile=False)
img0 = cv2.imread("a.jpg")
text0 = get_text(img0)
imgs0 = np.array(list(_get_imgs(img0)))
imgs0 = preprocess_input(imgs0)
word_model.predict(text0)
image_model.predict(imgs0)


@app.route('/verify', methods=['POST', 'GET'])
def hello_world():
    img = request.form['img']

    result = ''
    img = cv2.imdecode(np.fromstring(base64.b64decode(img), np.uint8), cv2.IMREAD_COLOR)
    text = get_text(img)
    imgs = np.array(list(_get_imgs(img)))
    imgs = preprocess_input(imgs)

    # 识别文字
    label = word_model.predict(text)
    label = label.argmax()
    fp = open('texts.txt', encoding='utf-8')
    texts = [text.rstrip('\n') for text in fp]
    text = texts[label]

    # list放文字
    titles = [text]

    position = []

    # 获取下一个词
    # 根据第一个词的长度来定位第二个词的位置
    if len(text) == 1:
        offset = 27
    elif len(text) == 2:
        offset = 47
    else:
        offset = 60
    text2 = get_text(img, offset=offset)
    if text2.mean() < 0.95:
        label = word_model.predict(text2)
        label = label.argmax()
        text2 = texts[label]
        titles.append(text2)

    # 加载图片分类器
    labels = image_model.predict(imgs)
    labels = labels.argmax(axis=1)

    for pos, label in enumerate(labels):
        if texts[label] in titles:
            position.append(pos + 1)

    # 没有识别到结果
    if len(position) == 0:
        return result
    result = position

    response = {
        'msg': "success",
        'result': result
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run()
