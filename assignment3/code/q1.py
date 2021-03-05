import numpy as np
import pandas as pd

# dataset source: https://www.kaggle.com/sulianova/cardiovascular-disease-dataset
np.random.seed(0)

df = pd.read_csv("mlbootcamp5_train.csv", nrows = None)#4000)
y_col = "cardio"
# Normalization reduce overflow
all_columns = df.columns
for col_name in all_columns:
    if y_col!=col_name:
        df[f"{col_name}"] = df[col_name]/df[col_name].mean()
x_cols = "age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active"
x_cols = x_cols.split(",")
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
# w = np.random.random([num_cols,1])
# let's use 0 as starting point.
w = np.zeros([num_cols,1])
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

    condition_positive = (y_test==1).sum()
    condition_negative = (y_test==0).sum()
    true_positive = ((y_test==1)&(y_hat==1)).sum()
    true_negative = ((y_test==0)&(y_hat==0)).sum()
    false_positive = ((y_test==0)&(y_hat==1)).sum()
    false_negative = ((y_test==1)&(y_hat==0)).sum()
    
    assert condition_positive+condition_negative==len(y_hat)
    assert true_positive+true_negative+false_positive+false_negative==len(y_hat)

    epsilon=1e-8
    tpr = true_positive/condition_positive
    tnr = true_negative/condition_negative
    ppv = true_positive/(true_positive+false_positive+epsilon)
    npv = true_negative/(true_negative+false_negative+epsilon)

    accuracy = (y_test==y_hat).sum() / len(y_hat)
    print(f"Accuracy {accuracy:.3f}")
    print(f"Sensitivity {tpr:.3f}, Specificity {tnr:.3f}\nPrecision {ppv:.3f}, Negative Predictive Value {npv:.3f}")
    print("")

# Test before training
print("Before training:")
test_model()
# Train
gamma = 1e-6
num_epochs = int(1e3)
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
    if i%int(num_epochs/5)==0:
        print(f"Train loss: {l_sum}")

# Test after training

print("After training:")
test_model()