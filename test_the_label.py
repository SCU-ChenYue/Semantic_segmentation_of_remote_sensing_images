from sklearn.preprocessing import LabelEncoder
from PIL import Image
from unet import Unet
import cv2


src = cv2.imread('data/labeltest/' + 'label.png')
print(src)
