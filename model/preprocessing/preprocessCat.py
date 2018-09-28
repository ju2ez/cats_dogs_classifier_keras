import os
import cv2
import numpy as np
import pandas as pd
def preprocess(self):
    pathdir = '/home/julian/Documents/Programming/GitHub/Machine Learning/cats_dogs_classifier/model/images/cats/'
    directory = os.fsencode('/home/julian/Documents/Programming/GitHub/Machine Learning/cats_dogs_classifier/model/images/cats/')
    i=0
    df = []
    print("Start preprocessing the cat images...")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            path = str (pathdir+filename)
           # print(path)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            i+=1
           # print('File: ', filename)
            width = 480
            height = 360 
            dim = (width, height)
            # resize image
            if img is  None :
                print("NoneType")
            else:
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
                df.append([np.array(resized),i-1])  
            continue
        else:
            continue
    from matplotlib import pyplot as plt
    #plt.imshow(df[0 ][0])
    #plt.show() #show one image to test if the array is have been feeded correctly
    return df


