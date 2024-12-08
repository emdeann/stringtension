import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def mean(lst):
    return sum(lst)/len(lst)

# Done in place
def remove_outliers(lst, factor):
    stdev = np.std(lst)
    av = mean(lst)
    for e in lst:
        if abs(e-av) > stdev*factor:
            lst.remove(e)

def find_frequency(video_path, n, fps):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Initialize variables
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frames_passed = 0
    passed_last_frame = False
    first_frame = True
    string_y = 0
    string_x = 0
    bars_width = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        # Setup to do on first frame of video
        if first_frame:
            # eliminate black bars from video
            h, w, _ = frame.shape
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, thresh = cv.threshold(gray, 75, 255, cv.THRESH_BINARY)
            _, thresh_low = cv.threshold(gray, 0.001, 255, cv.THRESH_BINARY)
            # Black bars will stay dark even with small threshold
            while thresh_low[h-10, bars_width] == 0: # search left side for end of black bar
                bars_width += 1
            w -= 2*bars_width
                
            # find height of string in video
            thresh = thresh[0:h, bars_width:frame.shape[1]-bars_width] 
            node_x = w//n if n > 1 else 15 # offset node if n=1 (avoid video boundary)
            string_y = h//4 # starting y-value for searching is h/4
            string_found = False
            while not string_found:
                string_y += 1
                string_found = thresh[string_y, node_x] != 0
            first_frame = False
        
        # Crop bars out of frame        
        frame = frame[0:h, bars_width:frame.shape[1]-bars_width] 
        string_x = w//(2*n) # search on antinodes 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 25, 255, cv.THRESH_BINARY)   
        if thresh[string_y, string_x] != 0 and not passed_last_frame: # Fires when pixel at vertical y, 
            frame[string_y-2:string_y+2, string_x-5:string_x+5] = (0, 0, 255)
            thresh[string_y-20:string_y+20, string_x-50:string_x+50] = 255
            passed_last_frame = True
            frames_passed += 1
        else:
            passed_last_frame = False
        # cv.imshow('frame', frame)
        # if cv.waitKey(1) == ord('q'):
        #     break
    cap.release()
    
    elapsed = total_frames/fps
    period = 2*(elapsed/frames_passed)
    print(1/period)
