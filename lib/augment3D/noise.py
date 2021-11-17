import numpy as np
import torch
import random
import torchio as tio

def gaussian_noise(img_numpy, mean=0, std=0.001):
    noise = np.random.normal(mean, std, img_numpy.shape)

    return img_numpy + noise


class Noise(object):
    def __init__(self, mean=0, std=0.001,type="gaussian"):
        if type=="gaussian":
            self.noise_func = tio.RandomNoise(mean=mean,std=std) 
        elif type == "blur":
            self.noise_func = tio.RandomBlur(std=std*20)
        elif type == "motion":
            self.noise_func = tio.RandomMotion(degrees=30,translation=5,num_transforms=int(std/0.065)+1)
        elif type == "spike":
            self.noise_func = tio.RandomSpike(num_spikes=2,intensity=std)
        elif type == "ghost":
            self.noise_func = tio.RandomGhosting(num_ghosts=1,intensity=std)
        else:
            raise NameError('No type available')
        self.mean = mean
        self.std = std

    def __call__(self, img_array):
        """
        Args:
            img_array_numpy (numpy): Array of images to be flipped.

        Returns:
            img_numpy (numpy):  flipped img.
        """
        img_array_copy = img_array.copy()
        for i in range(len(img_array)):
            img = np.expand_dims(img_array[i],axis=0)
            img_array_copy[i]= self.noise_func(img)
        return [np.squeeze(img) for img in img_array_copy]
