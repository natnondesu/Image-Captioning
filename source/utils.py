import numpy as np
import os, json
from collections import Counter
import os

def prepare_data(img_path, json_file, vocab_threshold):
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

    vocab = [word for word in word_freq.keys() if word_freq[word] > vocab_threshold]
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
    enc_caption_len = []
    img_list = []
    cap_index = []
    for idx, img in enumerate(image_split):
        for cap in caption_split[idx]:
            # Dont sample caption more than max_len
            if len(cap) > max_len:
                continue
            # Word tokenization
            enc_cap = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in cap] + \
                        [vocab['<end>']] + ([vocab['<pad>']] * (max_len - len(cap)))
            # Append to global list
            enc_caption_list.append(enc_cap)
            enc_caption_len.append(len(cap)+2)
            img_list.append(img)
            cap_index.append(idx)

    return img_list, enc_caption_list, enc_caption_len, cap_index

def get_caption_back(token, vocab):
    caption = []
    rev_vocab = dict(map(reversed, vocab.items()))
    removewords = ["<start>", "<end>", "<pad>"]
    
    for cap in token:
        # find stop index
        try:
            stop_idx = cap.index(2)
        except:
            stop_idx = len(cap)-1

        words = list(map(rev_vocab.get, cap[:stop_idx+1]))
        post_word = [word for word in words if word not in removewords]
        caption.append(post_word)

    return caption
