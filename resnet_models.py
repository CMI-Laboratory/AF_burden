import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add, Flatten, Reshape
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional, Multiply
from keras.models import Sequential, Model

def ResidualBlock(x, filters, kernel_size, strides):
  x_shortcut = x
  x = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x)
  x = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding='same')(x)
 
  out = Add()([x, x_shortcut])
  out = Activation('relu')(out)
  return out

def convolution_Block(x, filters, kernel_size, strides):
  x_shortcut = x
  
  x = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Dropout(0.2)(x)
  x = Conv1D(filters = filters, kernel_size = kernel_size, strides = 2, padding='same')(x)

  x_shortcut = Conv1D(filters = filters, kernel_size = kernel_size, strides = 2, padding='same')(x_shortcut)
  x_shortcut = BatchNormalization()(x_shortcut)
  x_shortcut = Activation('relu')(x_shortcut)
  x_shortcut = Dropout(0.2)(x_shortcut)

  out = Add()([x, x_shortcut])
  out = Activation('relu')(out)

  return out



def Resnet_18(input_shape):
  input = Input(input_shape)
  x = input
  x = Conv1D(filters = 64, kernel_size = 7, strides = 2, activation = 'relu', padding='same')(x)
  x = MaxPool1D(pool_size = 2, strides=2, padding='same')(x)

  x = ResidualBlock(x, 64, 3, 1)
  x = ResidualBlock(x, 64, 3, 1)

  x = convolution_Block(x, 128, 1, 1)
  x = ResidualBlock(x, 128, 3, 1)
   
  x = convolution_Block(x, 256, 1, 1)
  x = ResidualBlock(x, 256, 3, 1)

  x = convolution_Block(x, 512, 1, 1)
  x = ResidualBlock(x, 512, 3, 1)

  x = GlobalAveragePooling1D()(x)
  output = Dense(1, activation='sigmoid')(x)
  
  model = Model(input, output)
  return model

  
def Resnet_34(input_shape):
  input = Input(input_shape)
  x = input
  
  x = Conv1D(filters = 64, kernel_size = 7, strides = 2, activation = 'relu', padding='same')(x)
  x = MaxPool1D(pool_size = 2, strides=2, padding='same')(x)

  x = ResidualBlock(x, 64, 3, 1)
  x = ResidualBlock(x, 64, 3, 1)
  x = ResidualBlock(x, 64, 3, 1)

  x = convolution_Block(x, 128, 1, 1)
  x = ResidualBlock(x, 128, 3, 1)
  x = ResidualBlock(x, 128, 3, 1)
  x = ResidualBlock(x, 128, 3, 1)

  x = convolution_Block(x, 256, 1, 1)
  x = ResidualBlock(x, 256, 3, 1)
  x = ResidualBlock(x, 256, 3, 1)
  x = ResidualBlock(x, 256, 3, 1)
  x = ResidualBlock(x, 256, 3, 1)
  x = ResidualBlock(x, 256, 3, 1)

  x = convolution_Block(x, 512, 1, 1)
  x = ResidualBlock(x, 512, 3, 1)
  x = ResidualBlock(x, 512, 3, 1)

  x = GlobalAveragePooling1D()(x)
  output = Dense(1, activation='sigmoid')(x)
  
  model = Model(input, output)
  return model
