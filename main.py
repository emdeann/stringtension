import get_frequency
import joblib
from sklearn import svm
from skimage import io, color, transform

#model = joblib.load('model.p') # load pretrained model for n-prediction
img = io.imread('string.png')
img = color.rgb2gray(img[...,0:3])
img = transform.resize(img, (64, 64))
img = img.flatten()
