from network import *
import numpy as np

# training data
x_train = [np.random.rand(10,10,1)]
y_train = [np.random.rand(4,4,2)]

# network
net = Network()
net.add(ConvLayer((10,10,1), (3,3), 1))
net.add(ActivationLayer((8,8,1), tanh, tanh_prime))
net.add(ConvLayer((8,8,1), (3,3), 1))
net.add(ActivationLayer((6,6,1), tanh, tanh_prime))
net.add(ConvLayer((6,6,1), (3,3), 2))
net.add(ActivationLayer((4,4,2), tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.3)

# test
out = net.predict(x_train)
print("predicted = ", out)
print("expected = ", y_train)
