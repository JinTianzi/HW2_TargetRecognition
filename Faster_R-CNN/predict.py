import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from frcnn import FRCNN

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    frcnn = FRCNN()

    crop            = False # 是否在单张图片预测后对目标进行截取
    count           = False # 是否进行目标的计数

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            # 可视化一阶段proposal box
            proposal_box_image = frcnn.proposal_box_draw(image, crop = crop, count = count)
            proposal_box_image.show() 
            # 可视化最终检测结果
            image = Image.open(img)
            r_image = frcnn.detect_image(image, crop = crop, count = count)
            r_image.show()  