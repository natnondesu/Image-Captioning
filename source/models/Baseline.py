import numpy as np
from torch import nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, pretrained, last_layer_size, latent_size):
        super(Encoder, self).__init__()
        self.preNet = pretrained
        # Turn off gradient for pretrained model
        for param in self.preNet.parameters():
            param.requires_grad = False

        # Exclude top layer
        pre_modules = list(self.preNet.children())[:-1]
        self.Net = nn.Sequential(*pre_modules)
        self.latent1 = nn.Linear(last_layer_size, latent_size*2)
        self.latent2 = nn.Linear(latent_size*2, latent_size)
        self.batchnorm1 = nn.BatchNorm1d(latent_size*2)
        self.batchnorm2 = nn.BatchNorm1d(latent_size)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.Net(x)
        x = x.view(x.shape[0], -1)
        x = self.latent1(x)
        x = self.batchnorm1(self.silu(x))
        x = self.latent2(x)
        out = self.batchnorm2(x)
        return out
        
class Decoder(nn.Module):
    def __init__(self, word_emb_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        #self.num_layers = num_layers

        self.Word2Vec = nn.Embedding(self.vocab_size, self.word_emb_size)
        self.GRUCell = nn.GRUCell(self.word_emb_size, hidden_size)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device='cuda')

    def forward(self, img, cap, caplen):
        batch_size = img.shape[0]
        word_emb = self.Word2Vec(cap)
        decoder_length = (caplen).tolist()
        # Put img latent on top of caption sequence.
        x_cat = torch.cat((img.unsqueeze(1), word_emb[:, :-1, :]), dim=1)
        h = self.init_hidden(batch_size)
        predictions = torch.zeros(batch_size, max(decoder_length), self.vocab_size).cuda()
        for i in range(0, max(decoder_length)):
            # Teacher forcing goes here.
            h = self.GRUCell(x_cat[:, i, :], h[:])
            preds = self.linear(h) # (batch, size_vocab)
            predictions[:, i, :] = preds

        return predictions, decoder_length

    def inference(self, img, max_len=30):

        batch_size = img.shape[0]
        output = np.ones((batch_size, max_len))
        inputs = img
        h = self.init_hidden(batch_size)
        step = 0
        while True:
            h = self.GRUCell(inputs, h[:])
            out = self.linear(h)
            out = out.squeeze(1)
            tok = torch.argmax(out, dim=1)

            output[:, step] = tok.cpu().numpy()
            
            if step>=(max_len-1):
                # We reach maximum config, Stop prediction
                break

            step += 1
            inputs = self.Word2Vec(tok)
            
        return output
        

            



