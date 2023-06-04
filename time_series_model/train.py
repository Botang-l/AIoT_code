import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import device, get_val_loss, un_normalize_data, get_mape


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


def train(args, train_data, val_data):
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
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))

    save_model = {'model': best_model}
    torch.save(save_model, args.model_path)


## Test
def test(args, test_data, m, n):
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
    model = LSTM(args.input_size,
                 args.hidden_size,
                 args.num_layers,
                 args.output_size,
                 batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(args.model_path)['model'])
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
    print(f"y_shape: {len(y)}")
    print(f"pred_shape: {len(pred)}")
    print(f"y_value: {y[:10]}")
    print(f"pred_value: {pred[:10]}")
    print(f"mape: {get_mape(y, pred)}")
    get_plot(args.seq_len, y, pred)    # 顯示分析結果  ~get_plot
