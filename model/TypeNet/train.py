import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

#================ Variables ================#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 128
num_epochs = 500    # -> number of time the model will see whole dataset
epoch_log = 2

window_size = 16
learning_rate = 3e-4  # -> learning rate
adam_beta1 = 0.5 # -> beta1 for AdamW optimizer
adam_beta2 = 0.999 # -> beta2 (momentum) value for AdamW optimizer

#================ Methods ================#

@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()
    loss, ap_distance, an_distance, an_ap_diff = 0, 0, 0, 0

    for data in dataloader:
        data = data[0].to(device)
        anchor, positive, negative = torch.split(data, window_size, dim=1)
        _, _, _, loss_dict = model(anchor=anchor, positive=positive, negative=negative, calculate_loss=True)    

        loss += loss_dict["loss"].item()
        ap_distance += torch.mean(loss_dict["ap_distance"]).item()
        an_distance += torch.mean(loss_dict["an_distance"]).item()
        an_ap_diff += torch.mean(loss_dict["an_ap_diff"]).item()

    loss /= len(dataloader)
    ap_distance /= len(dataloader)
    an_distance /= len(dataloader)
    an_ap_diff /= len(dataloader)

    model.train()

    return loss, ap_distance, an_distance, an_ap_diff


def train_loop(model, train_dataloader, validation_dataloader, device=device):
    model.train()
    train_loss_list, train_ap_list, train_an_list, train_an_ap_diff_list = [], [], [], []
    val_loss_list, val_ap_list, val_an_list, val_an_ap_diff_list = [], [], [], []
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_loss, epoch_ap_distance, epoch_an_distance, epoch_an_ap_diff = 0, 0, 0, 0
        for index, data in enumerate(train_dataloader):
            # if data[0].shape[0] < batch_size:
            #     continue
            data = data[0].to(device)
            model.zero_grad()
            anchor, positive, negative = torch.split(data, window_size, dim=1)
            _, _, _, loss_dict = model(anchor=anchor, positive=positive, negative=negative, calculate_loss=True)

            epoch_loss += loss_dict["loss"].item()
            epoch_ap_distance += torch.mean(loss_dict["ap_distance"]).item()
            epoch_an_distance += torch.mean(loss_dict["an_distance"]).item()
            epoch_an_ap_diff += torch.mean(loss_dict["an_ap_diff"]).item()

            loss_dict["loss"].backward()
            optim.step()

            if index % (len(train_dataloader) // epoch_log) == 0:
              print(f"[Epoch: {epoch} / {num_epochs}][{index:4d}/{len(train_dataloader):4d}] Loss: {loss_dict['loss'].item():2.5f}, ANP difference: {torch.mean(loss_dict['an_ap_diff']).item():3.6f}")
          
        epoch_loss /= len(train_dataloader)
        epoch_ap_distance /= len(train_dataloader)
        epoch_an_distance /= len(train_dataloader)
        epoch_an_ap_diff /= len(train_dataloader)

        train_loss_list.append(epoch_loss)
        train_ap_list.append(epoch_ap_distance)
        train_an_list.append(epoch_an_distance)
        train_an_ap_diff_list.append(epoch_an_ap_diff)

        val_loss, val_ap_distance, val_an_distance, val_an_ap_diff = evaluate_model(model, validation_dataloader)

        val_loss_list.append(val_loss)
        val_ap_list.append(val_ap_distance)
        val_an_list.append(val_an_distance)
        val_an_ap_diff_list.append(val_an_ap_diff)        

        print(f"###### [Epoch: {epoch} / {num_epochs}] Train: Epoch loss: {epoch_loss:2.5f}, Epoch ANP difference: {epoch_an_ap_diff:3.5f}")
        print(f"###### [Epoch: {epoch} / {num_epochs}] Valid: Epoch loss: {val_loss:2.5f}, Epoch ANP difference: {val_an_ap_diff:3.5f}")
      
    return train_loss_list, train_ap_list, train_an_list, train_an_ap_diff_list, val_loss_list, val_ap_list, val_an_list, val_an_ap_diff_list




