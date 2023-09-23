import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def take_orhogonal(k):
    y = np.zeros(4)
    y[0] = +k[1]
    y[1] = -k[0]
    y[2] = -k[3]
    y[3] = k[2]

    return y


def calculate_region(line, points):
    if np.dot(line, points) > 0:
        return 1
    else:
        return -1


if __name__ == "__main__":

    data_num = 1001

    line1 = np.random.rand(4)
    line2 = take_orhogonal(line1)

    mu, sigma = 0, 0.1
    y = np.zeros(1200)
    e = np.random.normal(mu, sigma, 1200)

    w1 = 0.2
    w2 = -0.3
    w3 = 0.8
    w4 = -0.6

    w11 = 0.15
    w22 = -0.35
    w33 = 0.75
    w44 = -0.8

    w111 = -0.4
    w222 = 0.25
    w333 = 0.65
    w444 = 0.70

    w1111 = -0.28
    w2222 = 0.38
    w3333 = -0.70
    w4444 = -0.85

    # w1 = random.uniform(-1, 1)
    # w2 = random.uniform(-1, 1)
    # w3 = random.uniform(-1, 1)
    # w4 = random.uniform(-1, 1)

    # w11 = random.uniform(-1, 1)
    # w22 = random.uniform(-1, 1)
    # w33 = random.uniform(-1, 1)
    # w44 = random.uniform(-1, 1)

    # w111 = random.uniform(-1, 1)
    # w222 = random.uniform(-1, 1)
    # w333 = random.uniform(-1, 1)
    # w444 = random.uniform(-1, 1)

    # w1111 = random.uniform(-1, 1)
    # w2222 = random.uniform(-1, 1)
    # w3333 = random.uniform(-1, 1)
    # w4444 = random.uniform(-1, 1)

    # y_t+1 = a*y_t + b*y_t-1 + c*e_t + d*e_t-1 + e_t+1

    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    
    
    for i in range(1200):
        et_1 = np.random.normal(mu, sigma, 1)
        if i - 1 < 0:
            y[0] = 0 + 0 + 0 + 0 + et_1
        elif i - 2 < 0:
            y[1] = y[0] + 0 + 0 + 0 + et_1
        else:
            if (
                calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
                and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
            ):
                counter1 +=1
                y[i] = w1 * y[i - 1] + w2 * y[i - 2] + w3 * e[i - 1] + w4 * e[i - 2] + et_1

            elif (
                calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
                and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter2 +=1
                y[i] = w11 * y[i - 1] + w22 * y[i - 2] + w33 * e[i - 1] + w44 * e[i - 2] + et_1

            elif (
                calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
                and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter3 +=1
                y[i] = w111 * y[i - 1] + w222 * y[i - 2] + w333 * e[i - 1] + w444 * e[i - 2] + et_1
            else:
                counter4 +=1
                y[i] = w1111 * y[i - 1] + w2222 * y[i - 2] + w3333 * e[i - 1] + w4444 * e[i - 2] + et_1
    
    print(counter1)
    print(counter2)
    print(counter3)
    print(counter4)
    
    
    
    plt.scatter(np.arange(len(y[200:])), y[200:])
    plt.savefig(f"y_{data_num}.png")
    plt.cla()

    plt.scatter(np.arange(len(y[200:])), e[200:])
    plt.savefig(f"e_{data_num}.png")
    plt.cla()
    
    
    df = pd.DataFrame({"y": y, "e": e})

    df.to_csv("Dataset/syntetic/{}.csv".format(data_num))

    f = open("weights.txt", "a")
    f.write(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}\n")
    f.close()

    print("control")
