import numpy as np
input_red = np.array([
    [0, 156, 155, 156, 158, 158],
    [0, 153, 154, 157, 159, 159],
    [0, 149, 151, 155, 158, 159],
    [0, 146, 146, 149, 153, 158],
    [0, 145, 143, 143, 148, 158]
])
input_green = np.array([
    [0, 167, 166, 167, 169, 169],
    [0, 164, 165, 168, 170, 170],
    [0, 160, 162, 166, 169, 170],
    [0, 156, 156, 159, 163, 168],
    [0, 155, 153, 153, 158, 168]
])
input_blue = np.array([
    [0, 163, 162, 163, 165, 165],
    [0, 160, 161, 164, 166, 166],
    [0, 156, 158, 162, 165, 166],
    [0, 155, 155, 158, 162, 167],
    [0, 154, 152, 152, 157, 167]
])
kernel_red = np.array([
    [-1, -1, 1],
    [0,  1, -1],
    [0,  1,  1]
])

kernel_green = np.array([
    [1,  0,  0],
    [1, -1, -1],
    [1,  0, -1]
])

kernel_blue = np.array([
    [0,  1,  1],
    [0,  1,  0],
    [1, -1,  1]
])
def convolve_2d(image, kernel):
    kernel_size = kernel.shape[0]
    output_size = (image.shape[0] - kernel_size + 1, image.shape[1] - kernel_size + 1)
    output = np.zeros(output_size)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            region = image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    return output
output_red = convolve_2d(input_red, kernel_red)
output_green = convolve_2d(input_green, kernel_green)
output_blue = convolve_2d(input_blue, kernel_blue)
print(output_red, output_green, output_blue)
