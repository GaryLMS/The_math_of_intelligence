import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

PATH = 'diamonds.csv'
POINT_CHOOSE = 2500


def get_data(PATH):
    raw = pd.read_csv(PATH)
    #print(data.columns)
    #print(df.loc[:,['A','B']])
    data = raw.loc[:,['carat','x','price']]
    print('Data :', data)
    total_data_count = data.shape[0]
    print('Total data count :', total_data_count)
    cleaned = [list(data.loc[random.randint(0,total_data_count-1)]) for i in range(POINT_CHOOSE)]

    print(cleaned)
    print(cleaned[0])
    print(cleaned[0][1])

    #print(cleaned[1])
    #print([[round(float(c[0]), 2), c[1], c[2]] for c in cleaned])

    #return [[round(float(c[0]), 2), c[1], c[2]] for c in cleaned]
    return cleaned
def compute_error_for_line_given_points(w0,w1,w2,points):
    total_Error = 0
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        total_Error += (z - (w0 + w1*x + w2*y)) **2
    return total_Error/ float(len(points))

def step_gradients(w0_current,w1_current,w2_current,points):
    w0_gradient = 0
    w1_gradient = 0
    w2_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        #print('hello')
        #print(x)
        #print(y)
        x = round(float(x), 2)
        #y = round(float(y), 2)
        w0_gradient += 2 * (z - w0_current - w1_current*x - w2_current*y) * (-1)
        w1_gradient += 2 * (z - w0_current - w1_current*x - w2_current*y) * (-x)
        w2_gradient += 2 * (z - w0_current - w1_current*x - w2_current*y) * (-y)
    w0_gradient /= N
    w1_gradient /= N
    w2_gradient /= N

    w0_update = w0_current - 0.005*w0_gradient
    w1_update = w1_current - 0.005*w1_gradient
    w2_update = w2_current - 0.005*w2_gradient

    return [w0_update, w1_update, w2_update]

def weight_runner(points,initial_w0, initial_w1, initial_w2,num_iterations):
    w0 = initial_w0
    w1 = initial_w1
    w2 = initial_w2

    for i in range(num_iterations):
        w0, w1, w2 = step_gradients(w0,w1,w2,points)

    return [w0, w1, w2]

def run(points) :
    [initial_w0, initial_w1, initial_w2] = np.random.normal(size = [3])
    num_iterations = 200
    print("Starting gradient descent at w0 = {0}, w1 = {1}, w2 = {2} error = {3}".format(initial_w0, initial_w1, initial_w2, compute_error_for_line_given_points(initial_w0, initial_w1, initial_w2, points)))
    result_w0, result_w1, result_w2 = weight_runner(points, initial_w0, initial_w1, initial_w2, num_iterations)
    print("After {0} iterations w0 = {1}, w2 = {2}, w3 = {3} error = {4}".format(num_iterations, result_w0, result_w1, result_w2,compute_error_for_line_given_points(result_w0,result_w1,result_w2,points)))

if __name__ == '__main__':
    run(get_data(PATH))