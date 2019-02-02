from layer import Layer

# inherit from base class Layer
class ActivationLayer(Layer):
    # input_shape = (1,i)   i the number of input neurons
    def __init__(self, input_shape, activation, activation_prime):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
