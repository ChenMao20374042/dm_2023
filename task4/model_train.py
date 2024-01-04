import json
import random
from datetime import datetime

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd


next_roads = {}
relations = pd.read_csv('../data/rel.csv', index_col=0)
for _, relation in tqdm(relations.iterrows()):
    origin, destination = relation["origin_id"], relation["destination_id"]
    if origin in next_roads:
        next_roads[origin].append(destination)
    else:
        next_roads[origin] = [destination]

selected_cars = set()
jump_csv = pd.read_csv('../data/jump_task.csv', index_col=0)
for _, row in tqdm(jump_csv.iterrows()):
    selected_cars.add(row["entity_id"])

trip_id_2_car_id = {}
trips = pd.read_csv('../data/traj.csv', index_col=0)
for _, trip in tqdm(trips.iterrows()):
    trip_id_2_car_id[trip["traj_id"]] = trip["entity_id"]


trip_id2path = {}
with open('../data/fmmr.txt') as f:
    for idx, line in enumerate(tqdm(f)):
        if idx == 0:
            assert line.split(";")[6] == "cpath"
            continue
        cols = line.split(";")
        trip_id = int(cols[0])
        car_id = trip_id_2_car_id[trip_id]
        if car_id not in selected_cars:
            continue
        if len(cols[6]) == 0:
            continue
        trip_id2path[trip_id] = list(map(int, cols[6].split(",")))

random.seed(42)
trip_id2path = random.sample(trip_id2path.items(), 1000)
inputs_all = []
labels_all = []
for trip_id, path in tqdm(trip_id2path):
    car_id = trip_id_2_car_id[trip_id]
    path = [[road, car_id] for road in path]
    for idx in range(1, len(path)):
        inputs_all.append(torch.tensor(path[:idx], dtype=torch.float))
        labels_all.append(path[idx][0])
print(f'total trails: {len(trip_id2path)}')
print(f'total data items: {len(labels_all)}')
labels_all = torch.tensor(labels_all)

writer = SummaryWriter()


def get_config(name):
    with open('./config.json') as f_:
        config = json.load(f_)
    return config[name]


def load_data(batch_size_):
    inputs_train_, inputs_val_, labels_train_, labels_val_ = \
        train_test_split(inputs_all, labels_all, test_size=0.05, random_state=42)
    length_train_ = torch.tensor([len(inputs_train_item) for inputs_train_item in inputs_train_])
    length_val_ = torch.tensor([len(inputs_val_item) for inputs_val_item in inputs_val_])
    pad_inputs_train_ = pad_sequence(inputs_train_, batch_first=True, padding_value=-1)
    pad_inputs_val_ = pad_sequence(inputs_val_, batch_first=True, padding_value=-1)
    train_dataset_ = TensorDataset(pad_inputs_train_, labels_train_, length_train_)
    val_dataset_ = TensorDataset(pad_inputs_val_, labels_val_, length_val_)
    train_loader_ = DataLoader(dataset=train_dataset_, batch_size=batch_size_, shuffle=True)
    val_loader_ = DataLoader(dataset=val_dataset_, batch_size=batch_size_, shuffle=False)
    return train_loader_, val_loader_


def evaluate():
    model.eval()
    correct = total = 0
    print('evaluating...')
    with torch.no_grad():
        for inputs_, labels_, length_ in val_dataloader:
            outputs_ = model(inputs_, length_)
            for i in range(len(inputs_)):
                last_way = inputs_[i][length_[i] - 1][0].item()
                next_ways = next_roads[last_way]
                predict = outputs_[i][next_ways]
                predict_way = torch.argmax(predict)
                if next_ways[predict_way.item()] == labels_[i].item():
                    correct += 1
                total += 1
    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {accuracy * 100:.2f}%')
    if accuracy > get_config("accuracy line"):
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(model.state_dict(), f'models/{current_datetime}__{accuracy}.pth')
    model.train()
    return accuracy


class LSTM(nn.Module):
    def __init__(self, input_size_, hidden_size_, output_size_):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size_, hidden_size_, batch_first=True, dtype=torch.float)
        self.fc = nn.Linear(hidden_size_, output_size_, dtype=torch.float)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed_x)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :])
        return out


input_size = 2
hidden_size = 64
output_size = 38027

model = LSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
batch_size = 16
epochs = 20
for epoch in range(epochs):
    train_dataloader, val_dataloader = load_data(batch_size)
    bar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs} training')
    batch = 0
    model.train()
    for inputs, labels, length in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs, length)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bar.update(1)
        batch += 1
        if batch % 100 == 0:
            writer.add_scalar('Validation/Accuracy', evaluate(), epoch * len(train_dataloader) + batch)
    bar.close()
    scheduler.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Learning Rate: {optimizer.param_groups[0]["lr"]}')
