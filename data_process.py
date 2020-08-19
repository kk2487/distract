import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import matplotlib.pyplot as plt
import utils

length = 7

class DataProcess():

    train_npy = []
    
    def create_label(self, index):
        labels = [0] * 7
        labels[index]  = 1
        return np.array(labels)

    def create_training_data(self):
        for category in utils.CATEGORIES:
            path = os.path.join(utils.TRAIN_DATADIR, category)
            class_num = utils.CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
                    new_array = cv2.resize(img_array, (utils.IMG_SIZE, utils.IMG_SIZE))
                    self.train_npy.append([np.array(new_array), self.create_label(utils.CATEGORIES.index(category))])
                except Exception as e:
                    pass
        shuffle(self.train_npy)
        np.save(utils.TRAIN_SAVE_NPY, self.train_npy)


if __name__ == '__main__':
    train = DataProcess()
    train.create_training_data()
