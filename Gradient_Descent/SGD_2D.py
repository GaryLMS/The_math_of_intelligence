import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

PATH = 'diamonds.csv'

raw = pd.read_csv(PATH)

data = raw.as_matrix()

print(data)
print(type(data))
print(data.shape)


x_length = data[0:10000,8]
price = data[0:10000,7]
data_count = 10000


#mass = data[:,1]
#price = data[:,7]
#data_count = data.shape[0]

# plt.figure(num=1)
# plt.title('Diamond mass vs Price')
# plt.scatter(x=mass,y=price)
# plt.xlabel('mass')
# plt.ylabel('price')
# plt.show()

def total_MSE(m,b,x_length,price):
    total_error = 0

    for i in range(data_count):
        x = x_length[i]
        y = price[i]
        predict = m*x + b
        error = (y-predict) **2

        total_error += error

    return total_error/data_count

def Gradient_Descent(initial_m, initial_b,x_length,price,iteration):
    update_m = initial_m
    update_b = initial_b
    print('Start!, Iteration_count is : %d'%iteration)
    for k in range(iteration):
        m_gradient = 0
        b_gradient = 0
        for j in range(data_count):
            m_gradient += 2*(price[j] - update_m*x_length[j]- update_b)*(-x_length[j])
            b_gradient += 2*(price[j] - update_m*x_length[j]- update_b)*(-1)

        m_gradient /= data_count
        b_gradient /= data_count

        update_m -= learning_rate * m_gradient
        update_b -= learning_rate * b_gradient

        if(k % 100 == 0):
            error = total_MSE(update_m, update_b, x_length, price)
            print('At iteration %d, Error: %.6f'%(k+1,error))
    return update_m, update_b


[initial_m, initial_b] = np.random.normal(size = [2])
iteration = 2000
learning_rate = 0.001

fit_m, fit_b = Gradient_Descent(initial_m, initial_b, x_length, price, iteration)
print(fit_m,fit_b)
plt.figure(num=1)
plt.title('Diamond x_length vs Price')
plt.scatter(x=x_length,y=price)
plt.plot(x_length,fit_m*x_length+fit_b, color='red', label='Fit line')
plt.xlabel('x_length')
plt.ylabel('price')
#plt.show()

# Error surface

m = np.arange(500,700,10)
b = np.arange(-150,-180,-10)


X,Y = np.meshgrid(m,b)
zs = np.array([total_MSE(x,y,x_length,price) for x,y in zip(np.ravel(X),np.ravel(Y))])
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(X, Y, Z,cmap='hot')

ax.set_title('Gradient Descent')
ax.set_xlabel('slope (m)')
ax.set_ylabel('y-intercept (b)')
ax.set_zlabel('Error')

plt.show()