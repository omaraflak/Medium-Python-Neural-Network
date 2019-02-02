from network import *
import numpy as np

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer((1,2), (1,3)))
net.add(ActivationLayer((1,3), tanh, tanh_prime))
net.add(FCLayer((1,3), (1,1)))
net.add(ActivationLayer((1,1), tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
