import numpy as np
import os, json
from collections import Counter
import torch
import os

def prepare_data(img_path, json_file):
    # Read data split from json
    with open(json_file, 'r') as j:
        data = json.load(j)
        
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    word_freq = Counter()
    
    for img in (data['images']):
        caption = []
        for cap in img['sentences']:
            caption.append(cap['tokens'])
            word_freq.update(cap['tokens'])

        full_path = os.path.join(img_path, img['filename'])

        if img['split'] == 'train':
            train_x.append(full_path)
            train_y.append(caption)
        elif img['split'] == 'val':
            val_x.append(full_path)
            val_y.append(caption)
        elif img['split'] == 'test':
            test_x.append(full_path)
            test_y.append(caption)

    vocab = [word for word in word_freq.keys() if word_freq[word] > 3]
    vocab_dict = {key:value+4 for value, key in enumerate(vocab)}
    vocab_dict["<unk>"] = 3
    vocab_dict["<start>"] = 1
    vocab_dict["<end>"] = 2
    vocab_dict["<pad>"] = 0
    
    # Save vocab map if it need. (Maybe for inferences)
    #vocab_file_name = 'Flikr8k_vocab_mapping'
    #with open((vocab_file_name + '.json'), 'w') as j:
    #    json.dump(vocab_dict, j)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, vocab_dict 

def Create_image_caption_pair(image_split, caption_split, vocab, max_len=50):
    enc_caption_list = []
    img_list = []
    for idx, img in enumerate(image_split):
        for cap in caption_split[idx]:
            # Word tokenization
            enc_cap = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in cap] + \
                        [vocab['<end>']] + ([vocab['<pad>']] * (max_len - len(cap)))
            # Append to global list
            enc_caption_list.append(enc_cap)
            img_list.append(img)

    return img_list, enc_caption_list


def train(encoder, decoder, device, train_loader, optimizer, criterion, vocab_size):
    total_loss = 0
    encoder.train()
    decoder.train()
    for idx, data in enumerate(train_loader):
        data_img = data[0].to(device)
        data_cap = data[1].to(device)
        encoder.zero_grad()
        decoder.zero_grad()
        img_latent = encoder(data_img)
        output = decoder(img_latent, data_cap)    
        loss = criterion(output.view(-1, vocab_size), data_cap.view(-1).long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
 
    total_loss /= len(train_loader)

    return total_loss
        

# Test
def evaluate(encoder, decoder, device, test_loader, vocab_size):
    encoder.eval()
    decoder.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for data in test_loader:
            data_img = data[0].to(device)
            data_cap = data[1]
            img_latent = encoder(data_img)
            outputs = decoder.inference(img_latent)
            total_preds.append(outputs)
            total_labels.append(data_cap)

    return total_preds, total_labels