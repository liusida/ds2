import numpy as np
import pandas as pd

# dataset source: https://www.kaggle.com/macespinoza/mlbootcamp5/code
np.random.seed(0)

df = pd.read_csv("mlbootcamp5_train.csv", nrows = 4000)
# Normalization reduce overflow
df["height_n"] = df["height"]/df["height"].mean()
df["weight_n"] = df["weight"]/df["weight"].mean()

x_cols = "height_n,cholesterol,gluc,smoke,alco,active,cardio".split(",")
y_col = "weight_n"
num_cols = len(x_cols)

x = df[x_cols].to_numpy()
y = df[y_col].to_numpy()
num_record = y.shape[0]
num_train = int(.8 * num_record)
x_train = x[:num_train, :]
x_test = x[num_train:, :]
y_train = y[:num_train]
y_test = y[num_train:]

# weights: the thetas
w = np.random.random([num_cols,1])

# Test before training
y_hat = np.matmul(x_test, w)
l_mean_before = ((y_test-y_hat)**2).mean()
print(f"Test loss before training: {l_mean_before}")

# Train
gamma = 1e-6
num_epochs = int(1e4)
batch_size = 100
for i in range(num_epochs):
    for j in range(0, num_train, batch_size):
        x_batch = x_train[j:j+batch_size, :]
        y_batch = y_train[j:j+batch_size]

        y_hat = np.matmul(x_batch,w).flatten()
        diff = y_batch - y_hat

        gradient = -2 * np.matmul(x_batch.T, np.expand_dims(diff,-1))
        w -= gamma*gradient
    l_sum = (diff**2).sum()
    if i%int(num_epochs/10)==0:
        print(f"Train loss: {l_sum}")

# Test after training
y_hat = np.matmul(x_test, w)
l_mean_after = ((y_test-y_hat)**2).mean()
print(f"Test loss after training: {l_mean_after}")
print(f"Improved {l_mean_before-l_mean_after}")