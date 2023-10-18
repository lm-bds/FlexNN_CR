import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from dataprocess import DataPreprocessor
import pandas as pd
from model import DLearnModel 
import torch.nn as nn
import torch

import optuna
import plotly.express as px
from tune import objective
from loadin_data import load_data , create_final_val_loader, CustomDataset
from experiment import TensorboardExperiment
from runner import Runner, run_epoch
from synthesis import DataSynthesizer
import multiprocessing

# Determine the number of CPU cores
NUM_CORES = multiprocessing.cpu_count()




LOG_PATH = ".\\runs"
DATA_PATH = "~\\NN\\NN\\data\\LM.csv"
NUM_EPOCHS = 500
TRAILS = 50

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

preprocessor = DataPreprocessor(columns_to_encode, columns_to_drop)

data = preprocessor.load_data(file_path = DATA_PATH).cleaner().get_data()
data = data.fillna(0)
print(data)
df = pd.DataFrame()
pred_data = data.drop(columns=target_column)
df['actual'] = data["Cardiac_hospitalisations"]
data_synthesizer = DataSynthesizer(noise_percentage=1)
#data = data_synthesizer.generate_synthetic_data(data, num_synthetic_samples=10)
print(data)
input_size=data.shape[1]-1
print(input_size)
print(data.shape[1])
print(data.shape[0])
model = DLearnModel(input_size, hidden_units=[3])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

batch_size = 32

train_loader, val_loader, X_train, X_test, y_train, y_test = load_data(data, target_column, batch_size)

inputsize = data.shape[1] - len(target_column)



study = optuna.create_study(direction='minimize')
study.optimize(lambda trial : objective(trial, inputsize, train_loader, val_loader), n_trials=TRAILS, n_jobs=NUM_CORES)

best_params = study.best_params


best_hidden_layers = best_params['num_hidden_layers']
best_hidden_units = [best_params[f'hidden_units_layer_{i}'] for i in range(best_hidden_layers)]
best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_weight_decay = best_params['weight_decay']
best_batch_size = best_params["batch_size"]

train_loader, val_loader, _, _, _, _ = load_data(data, target_column, best_batch_size)


def main():
    
    best_model = DLearnModel(input_size=inputsize, hidden_units=best_hidden_units)
    best_optimizer = torch.optim.SGD(best_model.parameters(), lr=best_learning_rate,weight_decay=best_weight_decay)
    # Model and Optimizer

    best_model.train()
    # Create the runners
    test_runner = Runner(val_loader, best_model)
    train_runner = Runner(train_loader, best_model, best_optimizer)

    # Setup the experiment tracker
    tracker = TensorboardExperiment(log_path=LOG_PATH)

    # Run the epochs
    for epoch in range(NUM_EPOCHS):
        best_model.train()
        run_epoch(test_runner, train_runner, tracker, epoch)

        # Compute Average Epoch Metrics
        summary = ", ".join(
            [
                f"[Epoch: {epoch + 1}/{NUM_EPOCHS}]",
                f"Test loss: {test_runner.avg_loss: 0.4f}",
                f"Train loss: {train_runner.avg_loss: 0.4f}",
            ]
        )
        print("\n" + summary + "\n")

        # Reset the runners
        train_runner.reset()
        test_runner.reset()

        # Flush the tracker after every epoch for live updates
        tracker.flush()
    
    best_model.eval() 

    predictions = []

    best_model.eval()

    with torch.no_grad():
        for batch_x, batch_y in val_loader: 
            outputs = best_model(batch_x)
            actual = batch_y.detach().cpu().numpy()
    mse = mean_squared_error(actual, outputs)
    rmse = np.sqrt(mse)

    print(outputs)
    print(actual)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


    

    #





#print(model)

#print(data)


if __name__ == "__main__":
    main()


  
