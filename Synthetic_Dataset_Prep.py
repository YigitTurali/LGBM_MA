import numpy as np
import pandas as pd


def ARIMA(phi=np.array([0]), theta=np.array([0]), d=0, t=0, mu=0, sigma=1, n=20, burn=5, init=None):
    """ Simulate data from ARMA model (eq. 1.2.4):

    y_t = phi_1*y_{t-1} + ... + phi_p*y_{t-p} + theta_0*epsilon_t + theta_1*epsilon_{t-1} + ... + theta_q*epsilon_{t-q}

    with d unit roots for ARIMA model.

    Arguments:
    phi -- array of shape (p,) or (p, 1) containing phi_1, phi2, ... for AR model
    theta -- array of shape (q) or (q, 1) containing theta_1, theta_2, ... for MA model
    d -- number of unit roots for non-stationary time series
    t -- value deterministic linear trend
    mu -- mean value for normal distribution error term
    sigma -- standard deviation for normal distribution error term
    n -- length time series
    burn -- number of discarded values because series beginns without lagged terms

    Return:
    x -- simulated ARMA process of shape (n, 1)
    """
    # add theta_0 = 1 to theta
    theta = np.append(1, theta)

    # phi -- array of shape (p,)
    p = phi.shape[0]

    # theta -- array of shape (q,1)
    q = theta.shape[0]

    # add error terms (n+q)
    epsilon = np.random.normal(mu, sigma, (n + max(n, q) + burn, 1))

    # create array for returned values
    x = np.zeros((n + max(n, q) + burn, 1))

    # initialize first time series value y_{t-p}=theta_0*epsilon_{t-q}
    x[0] = epsilon[0]

    for i in range(1, x.shape[0]):
        AR = np.dot(phi[0: min(i, p)], np.flip(x[i - min(i, p): i], 0))
        MA = np.dot(theta[0: min(i + 1, q)], np.flip(epsilon[i - min(i, q - 1): i + 1], 0))
        x[i] = AR + MA + t

    # add unit roots
    if d != 0:
        ARMA = x[-n:]
        m = ARMA.shape[0]
        y = np.zeros((m + 1, 1))  # create temp array

        for i in range(d):
            for j in range(m):
                y[j + 1] = ARMA[j] + y[j]
            ARMA = y[1:]
        x[-n:] = y[1:]

    return x[-n:]


def synthetic_dataset(phi, theta, mu, sigma, dataset_length=1001):
    def take_orhogonal(k):
        y = np.zeros(4)
        y[0] = k[3]
        y[1] = -k[1]
        y[2] = -k[2]
        y[3] = k[4]

        return y

    def calculate_region(line, points):
        if np.dot(line, points) > 0:
            return 1
        else:
            return -1

    line1 = np.random.rand(4)
    line2 = take_orhogonal(line1)

    y = np.zeros(dataset_length)
    e = np.random.normal(mu, sigma, dataset_length)

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

    # y_t+1 = a*y_t + b*y_t-1 + c*e_t + d*e_t-1 + e_t+1

    counter1 = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0

    for i in range(dataset_length):
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
                counter1 += 1
                y[i] = w1 * y[i - 1] + w2 * y[i - 2] + w3 * e[i - 1] + w4 * e[i - 2] + et_1

            elif (
                    calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == 1
                    and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter2 += 1
                y[i] = w11 * y[i - 1] + w22 * y[i - 2] + w33 * e[i - 1] + w44 * e[i - 2] + et_1

            elif (
                    calculate_region(line1, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
                    and calculate_region(line2, np.array([y[i - 1], y[i - 2], e[i - 1], e[i - 2]])) == -1
            ):
                counter3 += 1
                y[i] = w111 * y[i - 1] + w222 * y[i - 2] + w333 * e[i - 1] + w444 * e[i - 2] + et_1
            else:
                counter4 += 1
                y[i] = w1111 * y[i - 1] + w2222 * y[i - 2] + w3333 * e[i - 1] + w4444 * e[i - 2] + et_1

    df = pd.DataFrame(y[200:])
    for i in range(phi):
        df[f"y-{i}"] = df["y"].shift(i)
    for j in range(theta):
        df[f"e-{j}"] = df["e"].shift(j)

    return df


def create_ARIMA_data(phi, theta, mu, sigma, t, n, test_size, ):
    y = ARIMA(phi=phi, theta=theta, mu=mu, sigma=sigma, n=n, t=t)
    y_train_new = y[:-test_size]
    y_test_new = y[-test_size:]
    return y_train_new, y_test_new


def create_dataset_mlp(y_train, y_test, lags, batch_size=100, val_ratio=0.2):
    for lag in lags:
        y_train["y" + '_lag_' + str(lag)] = y_train["y"].transform(lambda x: x.shift(lag, fill_value=0))
        y_test["y" + '_lag_' + str(lag)] = y_test["y"].transform(lambda x: x.shift(lag, fill_value=0))

    val_size = int(len(y_train) * val_ratio)
    if batch_size > 0:
        batch_indexes = np.random.choice(np.arange(len(y_train) - val_size), (64,))
    else:
        batch_indexes = np.arange(len(y_train) - val_size)

    X_train = y_train.drop("y", axis=1).values[:-val_size, :][batch_indexes, :]
    y_train = y_train[-1][:-val_size]
    X_val = y_train.drop("y", axis=1).values[-val_size:, :]
    y_val = y_train[-1][-val_size:]
    X_test = y_test.drop("y", axis=1).values
    y_test = y_test[-1]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
