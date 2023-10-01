import random

def step(sum):
    if sum > 0:
        return 1
    else:
        return 0

def train_perceptron(input, output, learning_rate, epochs):
    weights = [random.uniform(-1, 1) for k in range(len(input[0]))]
    error = 0
    for j in range(epochs):
        sum_error = 0
        for i in range(len(input)):
            net = 0
            for j in range(len(input[0])):
                net += input[i][j] * weights[j]
            predicted_output = step(net)
            # Cập nhật trọng số
            for j in range(len(input[0])):
                weights[j] += learning_rate*(output[i] - predicted_output) * input[i][j]
            # sai số
            error = error+0.5*pow((output[i] - predicted_output), 2)
            sum_error+=error
        if sum_error == 0:
            break
    return weights

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
output = [0, 1, 1, 1]

learning_rate = 0.1
epochs = 1000

weights = train_perceptron(input, output, learning_rate, epochs)

print("Trọng số cuối cùng:", weights)
print("Kết quả dự đoán:")
for i in range(len(input)):
    sum = 0
    for j in range(len(weights)):
        sum += input[i][j] * weights[j]
    predicted_output = step(sum)
    print("{}->{}".format(input[i], predicted_output))