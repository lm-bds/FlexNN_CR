
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix,mean_squared_error, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict,GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import matplotlib.pyplot as plt

import warnings
from typing import Callable
from sklearn.dummy import DummyRegressor
from sklearn.neural_network import MLPRegressor
import plotly.express as px

from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna

import plotly.express as px
from tune import objective
from loadin_data import load_data, create_final_val_loader, CustomDataset
from experiment import TensorboardExperiment
from runner import Runner, run_epoch
from dataprocess import DataPreprocessor
from synthesis import DataSynthesizer
import multiprocessing

# Determine the number of CPU cores
NUM_CORES = multiprocessing.cpu_count()
LOG_PATH = ".\\runs"
DATA_PATH = "~\\NN\\NN\\data\\LM.csv"
NUM_EPOCHS = 300

columns_to_drop = ["Patient_ID_hash","Commencement.Date","CommencementDep.Tool",
                   "CompletionGlucose.Type","CompletionGlucose","CompletionWaist",
                   "X6month.Dep.Tool...Other", "X12month.Dep.Tool",
                   "No..cardiac.presentations.to.ED_Baseline",
                   "No..non.cardiac.presentations.to.ED_Baseline",
                   "No..cardiac.presentations.to.ED_6months",
                 "No..non.cardiac.presentations.to.ED_6months",
                 "No..cardiac.presentations.to.ED_12months",
                 "No..non.cardiac.presentations.to.ED_12months"]
 
columns_to_encode = ["Gender","Indigenous","Employment.Type","Living.Type",
                     "Completed","Smoking.Risk","Alcohol.Risk","Cholesterol.Risk",
                     "Weight.Risk","BP.Risk","Blood.Sugar.Risk","Exercise.Risk",
                     "Depression.Risk","CommencementDiabetes","CommencementHypertension",
                     "CommencementFamily.history","CommencementAspirin.therapy",
                     "CommencementBeta.blocker","CommencementDepression","CompletionSmoking",
                     "CompletionDiabetes","CompletionHypertension","CompletionFamily.history",
                     "CompletionAspirin.therapy","CompletionOther.therapy",
                     "CompletionOther.therapy...Other","CompletionBeta.blocker",
                     "CompletionBeta.blocker...Other","CompletionLipid.therapy",
                     "CompletionLipid.therapy...Other","CompletionACE.ARB","CompletionACE.ARB...Other",
                     "X6month.Depression","X12month.Aspirin.therapy","X12month.Other.therapy",
                     "X12month.Beta.blocker","X12month.Lipid.therapy","X12month.ACE.ARB",
                     "X12month.Depression"
                     ]


target_column = ["Cardiac_hospitalisations"
]

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_units=None, activation_function='ReLU', dropout_rate=0.5):
        super(RegressionModel, self).__init__()
        
        if hidden_units is None:
            # Default hidden_units if not provided
            hidden_units = [64, 32]  # Modify the default architecture as needed

        # Define the layers and architecture of your model based on the hidden_units and activation_function
        self.layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_units[i]))
            else:
                self.layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            if activation_function == 'ReLU':
                self.layers.append(nn.ReLU())
            elif activation_function == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation_function == 'Sigmoid':
                self.layers.append(nn.Sigmoid())
            # Add dropout layer with the specified dropout_rate
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_units[-1], 1))  # Output layer
        
        # Create a Sequential model to stack the layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)




def validation_step(model, criterion, batch_x, batch_y):
    # Set the model in evaluation mode
    model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

    return loss.item()

def validate(model, criterion, val_loader):
    # Initialize validation loss
    val_loss = 0.0

    # Loop through the validation dataset
    for batch_x, batch_y in val_loader:
        # Perform a single validation step
        loss = validation_step(model, criterion, batch_x, batch_y)
        val_loss += loss

    # Calculate and return the average validation loss
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train_step(model, optimizer, criterion, batch_x, batch_y, clip_gradients=False, max_norm=None):
    # Set the model in training mode
    model.train()

    # Forward pass
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()

    if clip_gradients:
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()

    # Return loss and gradients (optional)
    if clip_gradients:
        return loss.item(), torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        return loss.item(), None

class CorrMatrix:
    def __init__ (self, data : pd.DataFrame):
        self.data = data
    
    def corr_matrix(self):
        correlation_matrix = self.corr()
        plt.figure(figsize=(12, 10))
        with sns.axes_style("white"):
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f",
                        annot_kws={"size": 8}, xticklabels=True, yticklabels=True, cbar=False)
            plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        
def calculate_rmse( predictions: pd.DataFrame, y_test: pd.Series) -> float:
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

def plot_predictions(model,X_test, y_test, title):
    y_pred = model.predict(X_test)
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real Values', 'y': 'Predicted Values'},
                     title=title)
    fig.update_layout(showlegend=False)
    fig.add_shape(type='line', x0=min(y_test), x1=max(y_test), y0=min(y_test), y1=max(y_test))
    fig.show()
    



preprocessor = DataPreprocessor(columns_to_encode, columns_to_drop)

data = preprocessor.load_data(file_path = DATA_PATH).cleaner().get_data()
data = data.fillna(0)

data_synthesizer = DataSynthesizer(data)
data = data_synthesizer.generate_synthetic_data(num_synthetic_samples=1, synthesis_columns=data.columns)
data = data.head(1000)
input_size=data.shape[1]-1
print(input_size)
print(data.shape[1])
print(data.shape[0])


input_size=data.shape[1]-1

model = RegressionModel(input_size, hidden_units=[3])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_loader, val_loader, X_train, X_test, y_train, y_test = load_data(data, target_column, 32)

inputsize = data.shape[1] - len(target_column)



study = optuna.create_study(direction='minimize')
study.optimize(lambda trial : objective(trial, inputsize, train_loader, val_loader), n_trials=50, n_jobs=NUM_CORES)

best_params = study.best_params


best_hidden_layers = best_params['num_hidden_layers']
best_hidden_units = [best_params[f'hidden_units_layer_{i}'] for i in range(best_hidden_layers)]
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_weight_decay = best_params['weight_decay']
best_batch_size = best_params["batch_size"]
train_loader, val_loader, _, _, _, _ = load_data(data, target_column, best_batch_size)

best_model = RegressionModel(input_size=input_size, hidden_units=best_hidden_units)
best_optimizer = torch.optim.SGD(best_model.parameters(), lr=best_learning_rate,weight_decay=best_weight_decay)


num_epochs = 600
for epoch in range(num_epochs):
    best_model.train() 

    epoch_loss = 0.0

    for batch_x, batch_y in train_loader:
        best_optimizer.zero_grad()
        outputs = best_model(batch_x)
        loss = criterion(outputs, batch_y.view(-1, 1))
        loss.backward()
        best_optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Train Loss: {avg_epoch_loss:.4f}')


best_model.eval() 
test_loss = 0.0
val_data = preprocessor.load_data(file_path = DATA_PATH).cleaner().get_data()
final_val_loader = create_final_val_loader(val_data, target_column=target_column, batch_size=best_batch_size, num_workers=0, transform=None)
with torch.no_grad():
    for batch_x, batch_y in val_loader:  # Change to test_loader if evaluating on the test set
        test_outputs = best_model(batch_x)
        test_loss += criterion(test_outputs, batch_y.view(-1, 1)).item()

avg_test_loss = test_loss / len(train_loader)  # Change to len(test_loader) if evaluating on the test set

print(f'Avg. Test Loss: {avg_test_loss:.4f}')

model.eval()

# Initialize a list to store the predictions
predictions = []

# Disable gradient computation during evaluation
with torch.no_grad():
    for batch_x, _ in final_val_loader:  # Assuming you have a test DataLoader
        # Forward pass to make predictions
        outputs = best_model(batch_x)
        predictions.extend(outputs.numpy())  # Append predictions to the list

# Convert the list of predictions to a NumPy array
predictions = pd.DataFrame(predictions)


mse = mean_squared_error(val_data[target_column].values, predictions)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
df = pd.DataFrame()

df["pred"] = predictions
df["actual"] =pd.DataFrame(val_data[target_column])

fig = px.scatter(x=df.actual, y=df.pred, labels={'x': 'Real Values (days)', 'y': 'Predicted Values (days)'},
                     title="Actual vs predicted time in hospital")
fig.add_shape(type='line', x0=min(df.actual), x1=max(df.actual), y0=min(df.actual), y1=max(df.actual))
fig.show()