import os
import sys
import tensorflow as tf
import numpy as np 
import tflearn
import utils
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class CNN():
    def model_definition(self):
        # Defination of Model 
        tf.reset_default_graph()
        convnet = input_data(shape=[None, utils.IMG_SIZE, utils.IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        #fully connected layer with relu activation
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 512, activation='relu')
        convnet = dropout(convnet, 0.8)

        #fully connected with softmax activation ( OUTPUT LAYER )
        convnet = fully_connected(convnet, 6, activation='softmax')#change 7classes
        convnet = regression(convnet, optimizer='adam', learning_rate= utils.LR, loss='categorical_crossentropy', name='targets')

        return convnet
		
		
    def train_model(self , train_data):
        """
        train_data = np.load(train_data, encoding="latin1")
        train = train_data[:26223]#training data
        validation = train_data[26223:32776]#validation data
        """
        train_data = np.load(train_data, encoding="latin1")
        train = train_data[:14000]#training data
        validation = train_data[14000:15770]#validation data
        
       
        x_train = np.array([i[0] for i in train]).reshape(-1, utils.IMG_SIZE, utils.IMG_SIZE, 3)
        y_train = [i[1] for i in train]

        x_validation = np.array([i[0] for i in validation]).reshape(-1, utils.IMG_SIZE, utils.IMG_SIZE, 3)
        y_validation = [i[1] for i in validation]

        convnet = self.model_definition()

        model = tflearn.DNN(convnet, tensorboard_dir=utils.TENSORBOARD_DIR, tensorboard_verbose=0)
		
        model.fit({'input': x_train}, {'targets': y_train}, n_epoch=1, validation_set=({'input': x_validation}, {'targets': y_validation}), snapshot_step=500, show_metric=True, run_id=utils.MODEL_NAME)
        model.save(utils.SAVE_PATH)

		

    def load_model(self):
        convnet = self.model_definition()
        model = tflearn.DNN(convnet, tensorboard_dir='./models/test/log', tensorboard_verbose=0)
        if os.path.isfile("./models/test/cnn.model.meta"):
            model.load('./models/test/cnn.model')
        return model		


if __name__ == '__main__':
    if(sys.argv[1] == '--train'):
        training_data = utils.TRAIN_DATA_PATH
        network = CNN()
        cnn_model = network.train_model(training_data)
