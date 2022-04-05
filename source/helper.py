import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch

def train(encoder, decoder, device, train_loader, optimizer, criterion):
    total_loss = 0
    encoder.train()
    decoder.train()
    for idx, data in enumerate(train_loader):
        data_img = data[0].to(device)
        data_cap = data[1].to(device)
        caption_length = data[2]
        encoder.zero_grad()
        decoder.zero_grad()
        img_latent = encoder(data_img)
        outputs = decoder(img_latent, data_cap)
        packed_output = pack_padded_sequence(outputs, caption_length, batch_first=True, enforce_sorted=False)
        packed_label = pack_padded_sequence(data_cap, caption_length, batch_first=True, enforce_sorted=False)  
        loss = criterion(packed_output[0], packed_label[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx%10 == 0:
            print(f"Train step [{idx} / {len(train_loader)}], loss : ", loss.cpu().detach().numpy().item())
 
    total_loss /= len(train_loader)

    return total_loss
        

# Test
def evaluate(encoder, decoder, device, test_loader, criterion):
    encoder.eval()
    decoder.eval()
    total_preds = []
    total_labels = []
    losses = 0
    with torch.no_grad():
        for data in test_loader:
            data_img = data[0].to(device)
            data_cap = data[1]
            #caption_length = data[2]
            img_latent = encoder(data_img)
            outputs = decoder.inference(img_latent)
            #packed_output = pack_padded_sequence(outputs.unsqueeze(1), caption_length, batch_first=True, enforce_sorted=False)
            #packed_label = pack_padded_sequence(data_cap, caption_length, batch_first=True, enforce_sorted=False)  
            #loss = criterion(packed_output[0], packed_label[0])
            #losses += loss.items()
            total_preds.append(outputs)
            total_labels.append(data_cap)
        #losses /= len(test_loader)

    return total_preds, total_labels

def LR_scheduler_with_warmup(optimizer, LR, epoch, warmup_epoch=0, scale=0.7, set_LR=0.001, interval_epoch=2):
    """Sets the learning rate to the initial LR decayed by 5% every interval epochs"""
    lr = LR
    if epoch < warmup_epoch:
        lr = np.around((set_LR/warmup_epoch) * (epoch+1), decimals=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
        
    elif (epoch % interval_epoch) == 0:
        lr = np.around(LR * scale, decimals=5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr