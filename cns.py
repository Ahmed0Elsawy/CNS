def tanh(x):
    return (2 / (1 + 2.718281828459045 ** (-2 * x))) - 1
def neural_network(x, w1, w2, b1, b2):
    h_input = [sum(x[i] * w1[i][j] for i in range(len(x))) + b1 for j in range(len(w1[0]))]
    h_output = [tanh(val) for val in h_input]
    y_input = sum(h_output[j] * w2[j][0] for j in range(len(w2))) + b2
    y_output = tanh(y_input)
    return y_output
x = [1.0, 0.5]
w1 = [[0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5]]
w2 = [[0.5],
      [0.5],
      [0.5]]
b1 = 0.5
b2 = 0.7
output = neural_network(x, w1, w2, b1, b2)
print("----->",output,"<-----")