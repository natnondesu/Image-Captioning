import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from nltk.translate.bleu_score import corpus_bleu
from source.models.Attention import Attention
from source.utils import get_caption_back

def train(encoder, decoder, device, train_loader, optimizer, criterion, log_interval=50):
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
        outputs, decoded_lengths, att = decoder(img_latent, data_cap, caption_length)
        packed_output = pack_padded_sequence(outputs, decoded_lengths, batch_first=True, enforce_sorted=False)
        packed_label = pack_padded_sequence(data_cap, decoded_lengths, batch_first=True, enforce_sorted=False)  
        loss = criterion(packed_output[0], packed_label[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if idx%log_interval == 0:
            print(f"Train step [{idx} / {len(train_loader)}], loss : ", loss.cpu().detach().numpy().item())
 
    total_loss /= len(train_loader)

    return total_loss
        

# Test
def evaluate(encoder, decoder, device, test_loader, vocab_dict, caption):
    encoder.eval()
    decoder.eval()
    total_preds = []
    total_labels = []
    total_bleu = []

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            data_img = data[0].to(device)
            data_cap = data[1].to(device)
            caption_length = data[2]
            caption_idx = data[3]
            img_latent = encoder(data_img)
            outputs, alphas = decoder.inference(img_latent)
            outputs = outputs.tolist()
            hypotheses = get_caption_back(outputs, vocab_dict)
            references = [caption[i] for i in caption_idx]
            
            bleu4 = corpus_bleu(references, hypotheses)
            total_bleu.append(bleu4)

    return np.array(total_bleu).mean()

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