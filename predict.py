#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import time

yolo = YOLO()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        image = image.convert('RGB')
        print(image.mode)
    except:
        print('Open Error! Try again!')
        continue
    else:
        start = time.time()
        r_image = yolo.detect_image(image)
        print(time.time()-start)
        r_image.show()
