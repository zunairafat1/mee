import numpy as np
import matplotlib.pyplot as plt
from keras.dataset import mnist
from keras.layer import Input, Dense
from keras.optimizer import Adam
from keras.models import Model

encoding_dim = 32