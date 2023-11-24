from task3.model import Trajformer
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


IN_DIM = 7
HID_DIM = 128
OUT_DIM = 1
GPU = 0
NUM_ATTN_HEADS = 16
NUM_LAYERS = 8
BATCH_SIZE = 128
num_epochs = 100

def mae(pred, label):
    return torch.mean(torch.abs(pred-label))

def mape(pred, label):
    return torch.mean(torch.abs(pred-label)/label)


device = torch.device("cuda:"+str(GPU))
#device = torch.device("cpu")

# load data
train_x_data = np.load("./data/train_x.npy")[...,6:]
train_x_data = torch.tensor(train_x_data, dtype=torch.float)
train_y_data = np.load("./data/train_y.npy")
avg = np.mean(train_y_data)
std = np.std(train_y_data)
train_y_data = (train_y_data - avg) / std
train_y_data = torch.tensor(train_y_data, dtype=torch.float)
train_x_data = train_x_data.to(device)
train_y_data = train_y_data.to(device)


train_dataset = TensorDataset(train_x_data, train_y_data)

val_x_data = np.load("./data/val_x.npy")[...,6:]
val_x_data = torch.tensor(val_x_data, dtype=torch.float)
val_y_data = np.load("./data/val_y.npy")
val_y_data = torch.tensor(val_y_data, dtype=torch.float)
val_x_data = val_x_data.to(device)
val_y_data = val_y_data.to(device)
val_dataset = TensorDataset(val_x_data, val_y_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# init model

model = Trajformer(in_dim=IN_DIM, hid_dim=HID_DIM, num_attn_heads=NUM_ATTN_HEADS, num_layers=NUM_LAYERS)
model.to(device)


loss_func = mae
#loss_func = nn.SmoothL1Loss()
optimizer = optim.Adam(filter(lambda p : p.requires_grad,model.parameters()),lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False) 


best_val_mae  = 10000
best_val_mape = 1000
best_val_model = None


for epoch in range(num_epochs): 

    # train
    total_train_mae = 0
    total_train_step = len(train_loader)
    model.train()
    loop = tqdm(enumerate(train_loader), total =len(train_loader))
    for step, (x,y_real) in loop:
        y_pred = model(x, pad_val=-1)   
        loss = loss_func(y_pred, y_real)  
        optimizer.zero_grad() 
        loss.backward()		
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        train_mae = loss.item()
        total_train_mae += train_mae
        loop.set_postfix(mae=train_mae)
    print(f"Epoch [{epoch}/{num_epochs}] Train MAE:{total_train_mae/total_train_step}")    

    # validation
    model.eval()
    total_val_mae = total_val_mape = 0
    total_val_step = len(val_loader)
    with torch.no_grad():
        for step, (x, y_real) in enumerate(val_loader):
            y_pred = model(x, pad_val=-1)
            y_pred = y_pred * std + avg
            val_mae = mae(y_pred, y_real)
            val_mape = mape(y_pred, y_real)
            total_val_mae += val_mae
            total_val_mape += val_mape
                
    print(f"Epoch [{epoch}/{num_epochs}] Val MAE:{total_val_mae/total_val_step} Val MAPE:{total_val_mape/total_val_step}") 
    if total_val_mae/total_val_step < best_val_mae:
        best_val_mae = total_val_mae/total_val_step
        best_val_mape = total_val_mape/total_val_step
        torch.save(model.state_dict(), './best_val_model.pth')
        
    print(f"BEST MAE:{best_val_mae} MAPE:{best_val_mape}")

model.eval()
param = torch.load('./best_val_model.pth')
model.load_state_dict(param)
test_x_data = np.load('./data/test_x.npy')[..., 6:]
test_x_data = torch.tensor(test_x_data, dtype=torch.float).to(device)
test_y_data = []
with torch.no_grad():
    num_batch = int(test_x_data.shape[0] // BATCH_SIZE) + 1
    for i in range(num_batch):
        x = test_x_data[i*BATCH_SIZE : min(test_x_data.shape[0], (i+1)*BATCH_SIZE)]
        y = model(x, pad_val=-1)
        y = y * std + avg
        test_y_data.append(y)
test_y_data = torch.concat(test_y_data, dim=0).detach().cpu().numpy()
np.save('./data/test_y.npy', test_y_data)




