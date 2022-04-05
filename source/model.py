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
    def __init__(self, word_emb_size, hidden_size, num_layers, vocab_size):
        super(Decoder, self).__init__()
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.Word2Vec = nn.Embedding(self.vocab_size, self.word_emb_size)
        self.gru = nn.GRU(self.word_emb_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_size), device='cuda')

    def forward(self, img, cap):
        word_emb = self.Word2Vec(cap)
        # Put img latent on top of caption sequence.
        x_cat = torch.cat((img.unsqueeze(1), word_emb[:, :-1, :]), dim=1)
        h = self.init_hidden(img.shape[0])
        x, h = self.gru(x_cat, h)
        out = self.linear(x)

        return out

    def inference(self, img):
        output = []
        inputs = img.unsqueeze(1)
        h = self.init_hidden(inputs.shape[0])
      
        while True:
            x, h = self.gru(inputs, h)
            out = self.linear(x)
            out = out.squeeze(1)
            tok = torch.argmax(out, dim=1)
            output.append(tok.cpu().numpy()[0].item())
            
            if (tok == 2) or (len(output)>40):
                # We found <end> token, Stop prediction
                break

            inputs = self.Word2Vec(tok)
            inputs = inputs.unsqueeze(1)
            
        return torch.Tensor(output)


            



