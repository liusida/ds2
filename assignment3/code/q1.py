import numpy as np
import pandas as pd

# dataset source: https://www.kaggle.com/macespinoza/mlbootcamp5/code
np.random.seed(0)

df = pd.read_csv("mlbootcamp5_train.csv", nrows = 4000)
# Normalization reduce overflow
df["height_n"] = df["height"]/df["height"].mean()
df["weight_n"] = df["weight"]/df["weight"].mean()

x_cols = "height_n,weight_n,cholesterol,gluc,alco,active,cardio".split(",")
y_col = "smoke"
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
def model(x):
    """A discrimitive model"""
    return np.matmul(x, w).flatten()

# Test
def test_model():
    """test the model with existing w"""
    y_hat = model(x_test)
    l_mean_after = ((y_test-y_hat)**2).mean()
    print(f"Test loss: {l_mean_after}")

    y_hat=(y_hat>0.5).astype(np.int)
    accuracy = (y_test==y_hat).sum() / len(y_hat)
    print(f"Accuracy {accuracy}")

    print("")

# Test before training
print("Before training:")
test_model()
# Train
gamma = 1e-5
num_epochs = int(1e4)
batch_size = 100
for i in range(num_epochs):
    for j in range(0, num_train, batch_size):
        x_batch = x_train[j:j+batch_size, :]
        y_batch = y_train[j:j+batch_size]

        y_hat = model(x_batch)
        diff = y_batch - y_hat

        gradient = -2 * np.matmul(x_batch.T, np.expand_dims(diff,-1))
        w -= gamma*gradient
    l_sum = (diff**2).sum()
    if i%int(num_epochs/10)==0:
        print(f"Train loss: {l_sum}")

# Test after training

print("After training:")
test_model()