import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo dữ liệu
X1 = np.array([[0, 0], [1, 1], [2, 2]])
X2 = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])
Y1 = np.ones((X1.shape[0], 1))
Y2 = -np.ones((X2.shape[0], 1))

# Kết hợp dữ liệu và nhãn
X = np.vstack((X1, X2))
Y = np.vstack((Y1, Y2))

# Khởi tạo trọng số và ngưỡng
w = np.zeros(X.shape[1])
b = 0

# Chọn tốc độ học
learning_rate = 0.1

# Số lần lặp
max_iterations = 1000

# Huấn luyện mạng perceptron
for iteration in range(max_iterations):
    misclassified = 0  # Số lượng dữ liệu được phân loại sai
    for i in range(X.shape[0]):
        prediction = np.sign(np.dot(w, X[i, :]) + b)  # Dự đoán có dấu

        # Cập nhật
        if prediction != Y[i]:
            w = w + learning_rate * Y[i] * X[i, :]
            b = b + learning_rate * Y[i]
            misclassified += 1

    # Kiểm tra
    if misclassified == 0:
        break

# Hiển thị kết quả
plt.title('Perceptron Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 1', 'Class 2'], loc='best')
plt.grid(True)

# Tách dữ liệu thành hai lớp để vẽ biểu đồ
class1 = X[Y[:, 0] == 1]
class2 = X[Y[:, 0] == -1]

plt.scatter(class1[:, 0], class1[:, 1], marker='o', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], marker='x', label='Class 2')
plt.legend(loc='best')

plt.show()
