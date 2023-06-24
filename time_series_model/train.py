import sys

sys.path.append(r".")
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]

        return pred

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.rnn(input_seq, (h_0))
        pred = self.fc(output[:, -1 ,:])

        return pred 

class TPA_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers,
                          batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers,
                          batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        # attention_weights(batch_size, seq_len, 1)
        attention_weights = self.attention(output)
        # weighted_output(batch_size, seq_len, hidden_size)
        weighted_output = attention_weights * output
        # aggregated_output(batch_size, hidden_size)
        aggregated_output = weighted_output.sum(dim=1)
        # pred(batch_size, output_size)
        pred = self.linear(aggregated_output)
        return pred

def un_normalize_data(y, pred, target_max, target_min):
    """
    Un-normalizes the given normalized data back to its original scale.

    Args:
        y: A numpy array representing the original target values.
        pred: A numpy array representing the predicted values.
        target_max: The maximum value of the target variable in the original scale.
        target_min: The minimum value of the target variable in the original scale.

    Returns:
        The un-normalized original target values (y) and the un-normalized predicted values (pred).

    """
    y, pred = np.array(y), np.array(pred)
    y = (target_max - target_min) * y + target_min
    pred = (target_max - target_min) * pred + target_min

    return y, pred


def get_mape(y, pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) between the target values and predicted values.

    Args:
        y: A numpy array representing the target values.
        pred: A numpy array representing the predicted values.

    Returns:
        The MAPE value.

    """

    return np.mean(np.abs((y - pred) / y))

# def get_plot(model_name, y, pred):
#     """
#     Plots the line chart of y and pred.

#     Args:
#         model_name: A string representing the name of the model.
#         y: A list or array of true values.
#         pred: A list or array of predicted values.

#     Returns:
#         None
#     """

#     num_plots = math.ceil(len(y) / 100)

#     for i in range(num_plots):
#         start = i * 100
#         end = min(start + 100, len(y))  # Adjust the end index to handle the last segment

#         fig = plt.figure()
#         # Plot y and pred line chart
#         x_range = range(start, end)
#         y_slice = y[start:end]
#         pred_slice = pred[start:end]
#         plt.plot(x_range[:len(y_slice)], y_slice, c='green', marker='*', ms=1, alpha=0.75, label='true')
#         plt.plot(x_range[:len(pred_slice)], pred_slice, c='red', marker='o', ms=1, alpha=0.75, label='pred')

#         plt.title("Result")
#         plt.xlabel("Data")
#         plt.ylabel("Value")
#         plt.legend()
#         plt.savefig(f'./time_series_model/result/{model_name}/part_result_{model_name}_{i}.png')
#     fig1 = plt.figure()
#     # 繪製 y 和 pred 的折線圖
#     x_range = range(len(y))
#     plt.plot(x_range, y, c='green', marker='*', ms=1, alpha=0.75, label='true')
#     plt.plot(x_range, pred, c='red', marker='o', ms=1, alpha=0.75, label='pred')

#     plt.title("result")
#     plt.xlabel("data")
#     plt.ylabel("value")
#     plt.legend()
#     plt.savefig(f'./time_series_model/result/LSTM_result_{model_name}.png')

def get_plot(model_name, y, pred):
    """
    Plots the line chart of y and pred.

    Args:
        model_name: A string representing the name of the model.
        y: A list or array of true values.
        pred: A list or array of predicted values.

    Returns:
        None
    """

    fig = plt.figure()
    # 繪製 y 和 pred 的折線圖
    x_range = range(len(y))
    plt.plot(x_range, y, c='green', marker='*', ms=1, alpha=0.75, label='true')
    plt.plot(x_range, pred, c='red', marker='o', ms=1, alpha=0.75, label='pred')

    plt.title("result")
    plt.xlabel("data")
    plt.ylabel("value")
    plt.legend()
    plt.savefig(f'./time_series_model/result/LSTM_result_{model_name}.png')


def get_val_loss(model, val_data, loss_function):
    """
    Computes the average validation loss for a given model and validation data.

    Args:
        model: The model for which to compute the validation loss.
        val_data: The validation data (a DataLoader object).
        loss_function: The loss function to compute the loss.

    Returns:
        The average validation loss.

    """

    model.eval()
    val_loss = []
    with torch.no_grad():
        for seq, label in val_data:
            seq, label = seq.to(device), label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, model_path, train_data, val_data):
    """
    Trains a model using the given training and validation data.

    Args:
        args: An object containing training arguments.
        train_data: The training data (a DataLoader object).
        val_data: The validation data (a DataLoader object).

    """

    model = LSTM(args.input_size,
                 args.hidden_size,
                 args.num_layers,
                 args.output_size,
                 batch_size=args.batch_size).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=True)
    min_epochs, min_val_loss, best_model = 5, float('inf'), None
    train_loss_history, val_loss_history = [], []

    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        model.train()
        for seq, label in train_data:
            seq, label = seq.to(device), label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = get_val_loss(model, val_data, loss_function)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model.state_dict()

        scheduler.step(val_loss)
        train_loss_history.append(np.mean(train_loss))
        val_loss_history.append(val_loss)
        if (epoch + 1) % 100 == 0:
            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    save_model = {'model': model}
    torch.save(save_model, model_path)


## Test
def test(args, model_name, model_path, test_data, m, n):
    """
    Tests a trained model using the given test data.

    Args:
        args: An object containing test arguments.
        test_data: The test data (a DataLoader object).
        m: The maximum value for unnormalization.
        n: The minimum value for unnormalization.

    """

    pred, y = [], []
    print('Loading Model...')
    # model = LSTM(args.input_size,
    #              args.hidden_size,
    #              args.num_layers,
    #              args.output_size,
    #              batch_size=args.batch_size).to(device)
    # model.load_state_dict(torch.load(model_path)['model'])
    loaded_model = torch.load(model_path)
    model = loaded_model['model']
    model.eval()

    print('Start predicting...')
    with torch.no_grad():
        for seq, label in tqdm(test_data):
            label = label.data.tolist()
            y.extend(label)
            seq = seq.to(device)
            y_pred = model(seq).data.tolist()
            pred.extend(y_pred)

    # Unnormalize data
    y, pred = un_normalize_data(y, pred, m, n)

    # Print results
    print(f"y_value: {y[:10]}")
    print(f"pred_value: {pred[:10]}")
    print(f"mape: {get_mape(y, pred)}")
    get_plot(model_name, y, pred)
