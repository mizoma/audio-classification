

import pickle
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2gray


class MelspectrogramInput():
    """ 
    Create an object with the following properties 

        - audio.data: data for training (nrows, 128x128)
        - audio.images: np.array to render data as an image (nrows, 128, 128)
        - audio.target: numeric value for labels (nrows, )
        - audio.target_names: descriptions for labels (nrows, )

    TODO: handle multilabel problem in the target_names / target

    """

    def __init__(self, type):

        self.type = type
        self.imagedir = f'data/mels_{type}.pkl'
        self.labelfile = f'data/{type}.csv'
        self.image_height = 128
        self.image_width = 128

        self.target_names = self.produce_target_names()
        self.target = self.produce_target()
        self.images = self.produce_images()
        self.data = self.produce_data()


    def produce_data(self):

        return self.images.reshape(
            (len(self.images), self.image_height * self.image_width)
            )


    def produce_images(self):

        orig_images = pickle.load(open(self.imagedir, 'rb'))
        images = np.empty((len(orig_images), self.image_height, self.image_width))
        
        for i in range(len(orig_images)):
            tmp = Image.fromarray(orig_images[i])
            tmp = tmp.resize((self.image_height, self.image_width))
            tmp = np.asarray(tmp)
            tmp = rgb2gray(tmp)

            images[i] = tmp

        return images


    def produce_target(self):

        target = np.empty((len(self.target_names),))
        unique_names = np.unique(self.target_names)

        for i in range(len(self.target_names)):
            index = np.where(unique_names==self.target_names[i])[0][0]
            target[i] = index

        return target


    def produce_target_names(self):

        target_names = pd.read_csv(self.labelfile)
        
        return target_names['labels'].values


    
