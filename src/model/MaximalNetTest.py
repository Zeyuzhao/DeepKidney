from model.MaximalNet import *
import numpy as np
pi = MaximalNetWrapper()

pi.load_checkpoint(name="weighted_convnet_epoch_19.pt", folder="wandb/run-20190821_131619-t00lpy2e/")

data = {
    "x": np.array([100, 0, 3, 4, 5]),
    "edge_index": np.array([[0, 1], [1, 2]])
}

print(pi.predict(data))