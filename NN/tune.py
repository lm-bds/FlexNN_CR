from model import DLearnModel 
import optuna
import torch
import torch.nn as nn

def objective(trial, input_size, train_loader, val_loader):
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 4)  
    hidden_units = [trial.suggest_int(f'hidden_units_layer_{i}', 32, 256,2) for i in range(num_hidden_layers)]
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    activation_function = trial.suggest_categorical('activation_function', ['ReLU', 'Tanh', 'Sigmoid'])

    # Create an instance of your PyTorch model with the selected hyperparameters
    model = DLearnModel(input_size=input_size, hidden_units=hidden_units, activation_function=activation_function)

    # Define your loss function and optimizer
    criterion = nn.MSELoss()
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Training loop
    num_epochs = 100  
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)


    # Validation loop
    val_loss = 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            val_outputs = model(batch_x)
            val_loss += criterion(val_outputs, batch_y.view(-1, 1)).item()

    avg_val_loss = val_loss / len(val_loader)
    model.train()
    # Return the validation loss as the objective value to minimize
    return avg_val_loss