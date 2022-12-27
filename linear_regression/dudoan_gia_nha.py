import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("price_house.csv", delimiter=',')

"Change data into array"
data = data.values

"x = Diện tích, y= Giá"
a = np.array([data[:,0]]).T
y = np.array([data[:,1]]).T
# print(A.shape)

"Create vector Ones"
vector_ones = np.ones((a.shape[0], a.shape[1]), dtype= np.int8)
# print(vector_ones.shape)

"Concatenate A, Ones"
A = np.concatenate((a, vector_ones), axis= 1)
# print(A)

"Tính vector a, b"
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot (y)
# print(x[0])

"Tạo ra các vị trí tọa độ x li ti trải dài từ điểm dữ liệu bắt đầu ->Kết thúc"
x_gd = np.array([a[0], a[-1]])
# print("x_gd", x_gd)
# b= np.array([[1,46]]).T
# print("b", b)

"y = ax+b"
y_gd = x[0][0]*x_gd+x[1][0]

"Kiểm tra giá"
x_test = 90
y_test = x[0][0]*x_test+x[1][0]
print(y_test)

"visual data"
fig = plt.figure("GD linear")
plt.plot(a, y, "ro")
plt.plot(x_gd, y_gd)
plt.show()
