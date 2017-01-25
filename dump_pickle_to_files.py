import pickle
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():
    train_folder = "./train/"
    test_folder = "./test/"
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)
    else:
        print("removing {}".format(train_folder))
        shutil.rmtree(train_folder)
        os.mkdir(train_folder)
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
    else:
        print("removing {}".format(test_folder))
        shutil.rmtree(test_folder)
        os.mkdir(test_folder)
    training_file = "./train.p"
    testing_file = "./test.p"
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    assert (X_train.shape[0] == y_train.shape[0]), "number of images!=number of labels."
    assert (X_train.shape[1:] == (32, 32, 3)), "Image dimention must be 32 x 32 x 3."

    paths_list = []
    y_list = []
    for i in range(len(y_train)):
        cv2.imwrite(train_folder + str(i) + ".jpg", X_train[i])
        paths_list.append(train_folder+str(i)+".jpg")
        y_list.append(y_train[i])
    data = {'path': paths_list,'y': y_list}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf="train_data.csv",index=False)

    paths_list = []
    y_list = []
    for i in range(len(y_test)):
        cv2.imwrite(test_folder + str(i) + ".jpg", X_test[i])
        paths_list.append(test_folder + str(i) + ".jpg")
        y_list.append(y_test[i])
    data = {'path': paths_list, 'y': y_list}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf="test_data.csv", index=False)


if __name__ == "__main__":
    main()
