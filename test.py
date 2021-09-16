import cv2
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N = 30
loss_array = numpy.array([0.3987, 0.3824, 0.3779, 0.3663, 0.3602, 0.3547, 0.3533, 0.3485, 0.3409, 0.3312,
                          0.3302, 0.3289, 0.3277, 0.3184, 0.3104, 0.3101, 0.2997, 0.2943, 0.2897, 0.2882,
                          0.2865, 0.2854, 0.2851, 0.2833, 0.2801, 0.2797, 0.2793, 0.2776, 0.2750, 0.2746])
val_loss_array = numpy.array([0.4067, 0.3957, 0.3818, 0.3772, 0.3704, 0.3613, 0.3588, 0.3527, 0.3496, 0.3573,
                              0.3402, 0.3390, 0.3379, 0.3254, 0.3233, 0.3202, 0.3185, 0.3152, 0.3097, 0.2981,
                              0.2934, 0.2902, 0.2891, 0.2843, 0.2819, 0.2802, 0.2799, 0.2791, 0.2850, 0.2831])

acc_array = numpy.array([0.4832, 0.5581, 0.6092, 0.6505, 0.6931, 0.7021, 0.7437, 0.7849, 0.7992, 0.8167,
                         0.8220, 0.8339, 0.8476, 0.8599, 0.8671, 0.8792, 0.8808, 0.8923, 0.8992, 0.9013,
                         0.9125, 0.9273, 0.9329, 0.9482, 0.9499, 0.9503, 0.9536, 0.9557, 0.9581, 0.9602])
val_acc_array = numpy.array([0.4537, 0.5379, 0.5981, 0.6631, 0.6987, 0.7390, 0.7486, 0.7724, 0.8053, 0.8203,
                             0.8020, 0.8138, 0.8264, 0.8379, 0.8461, 0.8587, 0.8608, 0.8723, 0.8799, 0.8844,
                             0.8935, 0.9061, 0.9127, 0.9262, 0.9270, 0.9309, 0.9342, 0.9347, 0.9352, 0.9388])

plt.plot(np.arange(0, N), loss_array, label="train_loss")
plt.plot(np.arange(0, N), val_loss_array, label="val_loss")
plt.plot(np.arange(0, N), acc_array, label="train_acc")
plt.plot(np.arange(0, N), val_acc_array, label="val_acc")
plt.title("Training Loss and Accuracy on U-Net Satellite Seg")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot0.png')

