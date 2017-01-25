import pickle
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import random
import numpy as np

def transform_image(img, ang_range, shear_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.
    A Random uniform distribution is used to generate different parameters for transformation
    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows = img.shape[0]
    cols = img.shape[1]
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, Trans_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))

    if (len(img.shape) == 2):
        img = img.reshape(img.shape[0], img.shape[1], 1)
    elif (len(img.shape) == 3):
        pass
    else:
        raise ValueError("Not Supported image shape!")
    return img


def jitter(samplesPerTrack, jitter_ratio=1):
    jitter_folder = "./jitter/"
    if not os.path.isdir(jitter_folder):
        os.mkdir(jitter_folder)
    else:
        print("In Jittering: removing {}".format(jitter_folder))
        shutil.rmtree(jitter_folder)
        os.mkdir(jitter_folder)
    max_freq = 0
    for i in samplesPerTrack.keys():
        if max_freq < len(samplesPerTrack[i]):
            max_freq = len(samplesPerTrack[i])
    print("In Jittering: max_freq = " + str(max_freq))
    print("In Jittering: Begin to generate jittered image and write to disk..")
    paths_list = []
    y_list = []
    total_jitter_cnt = 0
    for i in samplesPerTrack.keys():
        count = len(samplesPerTrack[i])
        if count < max_freq * jitter_ratio:
            numToInsert = int(max_freq * jitter_ratio - count)
            for j in range(numToInsert):
                file_name = jitter_folder + "jittered_track" + str(i) + "_#" + str(j) + ".jpg"
                transformed = transform_image(cv2.imread(random.choice(samplesPerTrack[i])), 10, 4, 4)
                cv2.imwrite(file_name, transformed)
                paths_list.append(file_name)
                y_list.append(i)
                total_jitter_cnt += 1
    data = {'path': paths_list, 'y': y_list}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf="jitter_data.csv", index=False)
    print("In Jittering: Finish writing jittered image to files. Total images generated: {}".format(total_jitter_cnt))

def cat(df_train):
    samplesPerTrack = {}
    for i in range(df_train.shape[0]):
        if (not df_train['y'].iloc[i] in samplesPerTrack.keys()):
            samplesPerTrack[df_train['y'].iloc[i]] = []
            samplesPerTrack[df_train['y'].iloc[i]].append(df_train['path'].iloc[i])
        else:
            samplesPerTrack[df_train['y'].iloc[i]].append(df_train['path'].iloc[i])
    return samplesPerTrack

def main():
    jitter_ratio = 1
    df_train = pd.read_csv("./train_data.csv", sep=",\s*", engine='python')
    samplesPerTrack = cat(df_train)
    jitter(samplesPerTrack=samplesPerTrack,jitter_ratio=1)

if __name__ == "__main__":
    main()



