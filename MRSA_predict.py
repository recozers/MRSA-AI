import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load in data

MRSA_rates_trust_month = pd.read_csv("MRSA_BY_MONTH.csv")
AB_PRES_MONTH = pd.read_csv("AB_PRES_MONTH.csv")
ECOLI_RATES_MONTH = pd.read_csv("ECOLI_RATES_MONTH.csv")
AB_INTRAVENOUS_MONTH = pd.read_csv("INTRAVENOUS_AB_MONTH.csv")

#filter data

def filter_data(df):
    """takes a dataframe and filters it into the data we want to use"""
    df = df[df["Area Type"] == 'NHS region']
    df = df[['Area Name', 'Value','Time period','Area Type','Indicator Name']]
    ind = df['Indicator Name'].iloc[0]
    df = df.rename(columns = {'Value':ind})
    df["Time period"] = pd.to_datetime(df["Time period"], format="%b %Y")
    df = df[df["Time period"] >= pd.Timestamp("2022-01-01")]
    df = df.drop(labels = ['Area Type','Indicator Name'], axis = 1)
    df = df.dropna()

    return df

MRSA_rates_trust_month = filter_data(MRSA_rates_trust_month)
AB_PRES_MONTH = filter_data(AB_PRES_MONTH)
ECOLI_RATES_MONTH = filter_data(ECOLI_RATES_MONTH)
AB_INTRAVENOUS_MONTH = filter_data(AB_INTRAVENOUS_MONTH)

#join data

data = (
    pd.merge(MRSA_rates_trust_month, AB_PRES_MONTH, on=["Area Name", "Time period"], how="inner")
    .merge(ECOLI_RATES_MONTH, on=["Area Name", "Time period"], how="inner")
    .merge(AB_INTRAVENOUS_MONTH, on=["Area Name", "Time period"], how="inner")
)

pred_data = data
pred_data = pred_data.sort_values(by=["Area Name", "Time period"])

columns_to_drop = []

for col in pred_data.columns:
    # Skip specific columns
    if col in ['Area Name', 'Time period', 'MRSA bacteraemia 12-month rolling case counts and rates, by acute trust and month']:
        continue
    
    pred_data[f"{col}-1"] = pred_data.groupby("Area Name")[col].shift(1)

    # Add the original column to the drop list
    columns_to_drop.append(col)

# Drop the desired columns
pred_data.drop(columns=columns_to_drop, inplace=True)
pred_data = pred_data.dropna()
#create winter dummy variable
pred_data["Winter_dummy"] = pred_data["Time period"].dt.month.isin([11, 12, 1, 2]).astype(int)

y = pred_data['MRSA bacteraemia 12-month rolling case counts and rates, by acute trust and month']
X = pred_data.drop(labels = ['Time period','Area Name', 'MRSA bacteraemia 12-month rolling case counts and rates, by acute trust and month'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
scaler = StandardScaler()
kf = KFold(n_splits=10, shuffle=True, random_state=0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Linear Model

model = LinearRegression()

mae_scores = []
# Cross val

X_train_scaled = np.array(X_train_scaled)
y_train = np.array(y_train)

for train_index, val_index in kf.split(X_train_scaled):
    # Split the data into train and test sets
    X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
    y_tr, y_val = y_train[train_index], y_train[val_index]
    
    # Train the model
    model.fit(X_tr, y_tr)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate the Mean Squared Error
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)

best_mse = np.mean(mae_scores)
col_dropped = None

mae_scores = []

for col in range(X_train_scaled.shape[1]):
    X_new = np.delete(X_train_scaled, col, axis = 1)
    for train_index, val_index in kf.split(X_train_scaled):
        # Split the data into train and test sets
        X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        # Train the model
        model.fit(X_tr, y_tr)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate the Mean Squared Error
        mae = mean_absolute_error(y_val, y_pred)
        mae_scores.append(mae)

    # Calculate average MSE across all folds
    average_mse = np.mean(mae_scores)
    if average_mse < best_mse:
        best_mse = average_mse
        col_dropped = col

#Transform data if dropped
if col_dropped is not None:
    print(col_dropped, "dropped")
    np.delete(X_train_scaled, col_dropped, axis = 1)
    X_test_scaled.drop(labels = [col_dropped])

#fit model to full training set
model.fit(X_train_scaled, y_train)

y_hat = model.predict(X_test_scaled)

mae = mean_absolute_error(y_hat, y_test)

coefficients = model.coef_
column_names = X.columns
coef_dict = dict(zip(column_names, coefficients))
print("MAE Linear model = " + str(mae))
print()
print(coef_dict)
print()

#deep model

class DeepModel(nn.Module):
    """Deep Learning network with two hidden layers one of which is fully connected"""
    def __init__(self, hidden_size):
        super(DeepModel, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()  # Activation function
        self.fc3 = nn.Linear(hidden_size, 1)  # Output layer 

    def forward(self, x):
        """Forward pass through the network"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # No activation on the output
        return x

hidden_sizes = [25,50,75,100,150]
lrs = np.arange(0.001,0.1, 0.01)
best_mae = np.inf 

#Cross validation

for lr in lrs:
    for hidden_size in hidden_sizes:
        maes = []

        for train_index, val_index in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_index], X_train_scaled[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            X_tr = torch.tensor(X_tr, dtype=torch.float32)
            y_tr = torch.tensor(y_tr, dtype=torch.float32)

            # Prepare data loader

            dataset = TensorDataset(X_tr, y_tr)
            dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

            model = DeepModel(hidden_size)

            criterion = nn.MSELoss() 
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Training loop
            num_epochs = 100
            for epoch in range(num_epochs):
                for batch_X, batch_y in dataloader:
                    # Forward pass
                    outputs = model(batch_X)
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, batch_y)

                    # Backward pass and optimisation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            #Evaluate model

            model.eval()

            X_val = torch.tensor(X_val, dtype=torch.float32)

            with torch.no_grad():
                y_hat = model(X_val)

            mae = mean_absolute_error(y_val, y_hat)

            maes.append(mae)

            avg_mae = np.mean(maes)

            if avg_mae<best_mae:
                best_mae = avg_mae
                opt_lr = lr
                opt_hs = hidden_size

print("opt lr", opt_lr, "opt hs", opt_hs)
print()

#retrain on full training set

X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

dataset = TensorDataset(X_train_scaled, y_train)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

model = DeepModel(opt_hs)

criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=opt_lr)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#Evaluate model
model.eval()

X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)

with torch.no_grad():
    y_hat = model(X_test_scaled)

mae = mean_absolute_error(y_test, y_hat)

print("MAE deep learning model = " + str(mae))

#train on full dataset

X = X.to_numpy()
y = y.to_numpy()

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

model = DeepModel(opt_hs)

criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=opt_lr)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        outputs = outputs.squeeze(-1)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#inference

def predict(sample_data):
    with torch.no_grad():
        test_tensor = torch.tensor(sample_data,dtype=torch.float32)
        return model(test_tensor)

sample_data = [0.5,0.5,0.5,1]
prediction = predict(sample_data).detach().item()

print("Predicted MRSA Rate:", prediction)