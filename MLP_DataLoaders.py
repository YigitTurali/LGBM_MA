import numpy as np
import torch
import torch.utils.data as data_utils


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


def create_pytorch_loader(X, y, batch_size=100, shuffle=True):
    patterns = torch.from_numpy(X.astype('float32'))
    targets = torch.from_numpy(y.astype('float32'))

    dataset = data_utils.TensorDataset(patterns, targets)
    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_pytorch_data(y_train, y_test, batch_size=100):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset_mlp(y_train, y_test)
    train_loader = create_pytorch_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_pytorch_loader(X_val, y_val, batch_size=batch_size * 2, shuffle=False)
    test_loader = create_pytorch_loader(X_test, y_test, batch_size=batch_size * 2, shuffle=False)

    return train_loader, val_loader, test_loader
