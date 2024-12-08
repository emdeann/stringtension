import numpy as np
import sys
from get_frequency import find_frequency
import joblib
from sklearn import svm
from skimage import io, color, transform


def calculate_tension():
    model = joblib.load('standing_wave_classifier.pkl') # load pretrained model for n-prediction
    
    image_path = sys.argv[1]
    img = io.imread(image_path)
    img = color.rgb2gray(img[...,0:3])
    img = transform.resize(img, (64, 64))
    img = img.flatten()
    arr = np.array([img]).reshape(1,-1)
    n = model.predict(arr)[0] + 1# get first (only) element from model prediction based on image
    print(f'Wave harmonic: {n}')

    video_path = sys.argv[2]
    video_framerate = 304
    f = find_frequency(video_path, n, video_framerate)
    print(f'Calculated Frequency: {f} Hz')    

    # --- Final Calculation
    L = 2.0 # m
    m = 6.11e-4 # kg/m

    T = 4 * ((L * (f/n)) ** 2) * m
    print(f'Tension in the string: {T} N')


if len(sys.argv) == 3:
    calculate_tension()
else:
    print('Incorrect arguments passed. Calls should look like `st imgname.ext videoname.ext`')
    