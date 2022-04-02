import numpy as np
import os, json
from collections import Counter


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
    vocab_dict = {key:value+1 for value, key in enumerate(vocab)}
    vocab_dict["<unk>"] = len(vocab)+1
    vocab_dict["<start>"] = len(vocab)+1
    vocab_dict["<end>"] = len(vocab)+1
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