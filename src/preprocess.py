#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:11:25 2020

@author: neelkanthkundu
Preprocess the data to make it compatible for feedding it into the CNN model
"""
import h5py
import numpy as np

def get_preprocess_data(data_file):
  #This function reads the data from the file and pre processes it to make it compatible to be fed into 
  #the CNN: H (shape: [#samples,56,925,2]) is the input to the CNN and Pos (shape: [#samples,3] ) is the output
  #Input: data_file: path of the data file
  f = h5py.File(data_file, 'r')
  H_Re1 = f['H_Re'][:] #shape (sample size, 56, 924, 5) 
  H_Im1 = f['H_Im'][:] #shape (sample size, 56, 924, 5) 
  SNR1 = f['SNR'][:] #shape (sample size, 56, 5)
  Pos1 = f['Pos'][:] #shape(sample size, 3)
  f.close()
  H_Re=np.zeros([H_Re1.shape[0]*H_Re1.shape[-1],H_Re1.shape[1],H_Re1.shape[2]])
  H_Im=np.zeros([H_Im1.shape[0]*H_Im1.shape[-1],H_Im1.shape[1],H_Im1.shape[2]])
  SNR=np.zeros([SNR1.shape[0]*SNR1.shape[-1],SNR1.shape[1]])

  for i in range(5): #For moving the last axis of 5 measurements into the first axis
    H_Re[i*H_Re1.shape[0]:(i+1)*H_Re1.shape[0],:,:]  =H_Re1[:,:,:,i]
    H_Im[i*H_Im1.shape[0]:(i+1)*H_Im1.shape[0],:,:]  =H_Im1[:,:,:,i]
    SNR[i*SNR1.shape[0]:(i+1)*SNR1.shape[0],:]=SNR1[:,:,i]
    Pos=np.repeat(Pos1,5,axis=0)

  H1=np.zeros([H_Re.shape[0],H_Re.shape[1],H_Re.shape[2]+1])
  H1[:,:,0:924]=H_Re
  H1[:,:,924]=SNR
  H2=np.zeros([H_Im.shape[0],H_Im.shape[1],H_Im.shape[2]+1])
  H2[:,:,0:924]=H_Im
  H2[:,:,924]=SNR
  H=np.zeros([H1.shape[0],H1.shape[1],H1.shape[2],2])
  H[:,:,:,0]=H1
  H[:,:,:,1]=H2
  

  del H_Re1, H_Im1, SNR1, H_Re, H_Im, SNR, H1, H2, Pos1

  return H, Pos

"""
example usage:
    CTW_labelled ="../data/CTW2020_labelled_data/"
    data_file = CTW_labelled+"file_"+str(1)+".hdf5"
    H,Pos=get_preprocess_data(data_file)
"""




