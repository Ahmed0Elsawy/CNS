import math
import random
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]
        self.output_weights = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
        self.hidden_bias = [0.0 for _ in range(hidden_size)]
        self.output_bias = [0.0 for _ in range(output_size)]
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def forward_propagation(self, inputs):
        inputs = inputs[0] if isinstance(inputs[0], list) else inputs
        inputs = [float(x) for x in inputs]
        hidden_layer = [0.0 for _ in range(self.hidden_size)]
        for h in range(self.hidden_size):
            total = self.hidden_bias[h]
            for i in range(self.input_size):
                total += inputs[i] * self.hidden_weights[i][h]
            hidden_layer[h] = self.sigmoid(total)
        outputs = [0.0 for _ in range(self.output_size)]
        for o in range(self.output_size):
            total = self.output_bias[o]
            for h in range(self.hidden_size):
                total += hidden_layer[h] * self.output_weights[h][o]
            outputs[o] = self.sigmoid(total)
        return hidden_layer, outputs
    def backward_propagation(self, inputs, expected_outputs, learning_rate):
        inputs = inputs[0] if isinstance(inputs[0], list) else inputs
        expected_outputs = expected_outputs[0] if isinstance(expected_outputs[0], list) else expected_outputs
        inputs = [float(x) for x in inputs]
        expected_outputs = [float(x) for x in expected_outputs]
        hidden_layer, predicted_outputs = self.forward_propagation(inputs)
        output_errors = [
            (expected_outputs[o] - predicted_outputs[o]) *
            self.sigmoid_derivative(predicted_outputs[o])
            for o in range(self.output_size)
        ]
        hidden_errors = [0.0 for _ in range(self.hidden_size)]
        for h in range(self.hidden_size):
            error = 0.0
            for o in range(self.output_size):
                error += output_errors[o] * self.output_weights[h][o]
            hidden_errors[h] = error * self.sigmoid_derivative(hidden_layer[h])
        original_output_weights = [row[:] for row in self.output_weights]
        original_hidden_weights = [row[:] for row in self.hidden_weights]
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                self.output_weights[h][o] += learning_rate * output_errors[o] * hidden_layer[h]
        for o in range(self.output_size):
            self.output_bias[o] += learning_rate * output_errors[o]
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                self.hidden_weights[i][h] += learning_rate * hidden_errors[h] * inputs[i]
        for h in range(self.hidden_size):
            self.hidden_bias[h] += learning_rate * hidden_errors[h]
        print("Updated Output Weights:")
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                print(
                    f"From Hidden Neuron {h} to Output Neuron {o}: {original_output_weights[h][o]} -> {self.output_weights[h][o]}")
        print("\nUpdated Hidden Weights:")
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                print(
                    f"From Input Neuron {i} to Hidden Neuron {h}: {original_hidden_weights[i][h]} -> {self.hidden_weights[i][h]}")
        return sum((expected_outputs[o] - predicted_outputs[o]) ** 2 for o in range(self.output_size)) / 2
    def train(self, inputs, expected_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            error = self.backward_propagation(inputs, expected_outputs, learning_rate)
    def predict(self, inputs):
        _, outputs = self.forward_propagation(inputs)
        return outputs
def main():
    inputs = [0.05, 0.10]
    expected_outputs = [0.01, 0.99]
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
    nn.train([inputs], [expected_outputs], epochs=1, learning_rate=0.5)
if __name__ == "__main__":
    main()
