from this import d
from turtle import forward
import numpy as np
from torch import nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, pretrained, embedding_dim, latent_pix_dim=16):
        super(Encoder, self).__init__()
        self.preNet = pretrained
        # Turn off gradient for pretrained model
        for param in self.preNet.parameters():
            param.requires_grad = False

        # Exclude top layer
        pre_modules = list(self.preNet.children())[:-2]
        self.Net = nn.Sequential(*pre_modules)
        self.dense = nn.Linear(1408, embedding_dim)
        #self.pooling = nn.AdaptiveAvgPool2d(latent_pix_dim)

    def forward(self, x):
        x = self.Net(x) # output from pretrained (Batch_size, channel, pix, pix) --> In this case (Batch, 1408, 8, 8)
        x = x.reshape((x.shape[0], x.shape[1], -1)) # (Batch_size, channel, pix*pix)
        x = torch.moveaxis(x, 1, -1) # (Batch_size, pix*pix, channel)
        out = self.dense(x)
        #out = self.pooling(x) # (batch, channel, latent_pix_dim, latent_pix_dim)
        return out

class Attention(nn.Module): # Bahdanau
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_feat, decoder):
        # Encoder attention should be performed on channel dim, not pixel itself.
        # (Batch_size, pixel, embedding_dim)
        att1 = self.encoder_att(encoder_feat)
        att2 = self.decoder_att(decoder)
        fatt = self.relu(att1+att2.unsqueeze(1)) # Additive attention (Bahdanau)
        att = self.full_att(fatt) # (batch, pixel, 1)
        alpha = self.softmax(att)
        context_vector = torch.sum(alpha*encoder_feat, dim=1) # (batch, encode_channel, 1)
        return context_vector, alpha.squeeze()

        
class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, attention_dim, word_emb_size, decoder_hidden, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.word_emb_size = word_emb_size
        self.hidden_size = decoder_hidden
        self.vocab_size = vocab_size
        #self.num_layers = num_layers

        self.attention = Attention(encoder_dim, decoder_hidden, attention_dim)  # attention network
        self.Word2Vec = nn.Embedding(self.vocab_size, self.word_emb_size)
        #self.gru = nn.GRU(self.word_emb_size, decoder_hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.GRUCell = nn.GRUCell(word_emb_size+encoder_dim, decoder_hidden)
        self.linear = nn.Linear(decoder_hidden, self.vocab_size)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device='cuda')

    def forward(self, img, cap, caplen):
        batch_size = img.shape[0]
        num_pixel = img.shape[1]
        word_emb = self.Word2Vec(cap)
        h = self.init_hidden(img.shape[0])
        # Put img latent on top of caption sequence.
        decoder_length = (caplen).tolist() # To find what is the maximum length in this batch.
        predictions = torch.zeros(batch_size, max(decoder_length), self.vocab_size).cuda()
        alphas = torch.zeros((batch_size, max(decoder_length), num_pixel)).cuda()
        # Should be implemented GRUcell manually
        for i in range(1, max(decoder_length)):
            # Calculate Attention scores.
            context_vector, alpha = self.attention(img[:], h)
            # Teacher forcing goes here.
            h = self.GRUCell(torch.cat((word_emb[:, i-1, :], context_vector), dim=1), h[:])
            preds = self.linear(h) # (batch, size_vocab)
            predictions[:, i, :] = preds
            alphas[:, i, :] = alpha

        predictions[:, 0, 1] = 1 # Mark at <start> token
        return predictions, decoder_length, alphas

    def inference(self, img, max_len=50):
        batch_size = img.shape[0]
        num_pixel = img.shape[1]
        output = []
        alphas = []
        inputs = self.Word2Vec(torch.LongTensor([1]).cuda())
        h = self.init_hidden(img.shape[0])
        
        while True:
            context_vector, alpha = self.attention(img[:], h)
            h = self.GRUCell(torch.cat((inputs, context_vector), dim=1), h)
            out = self.linear(h)
            out = out.squeeze(1)
            tok = torch.argmax(out, dim=1)
            output.append(tok.cpu().numpy()[0].item())
            alphas.append(alpha.cpu().numpy())
            
            if (tok == 2) or (len(output)>=max_len):
                # We found <end> token, Stop prediction
                break

            inputs = self.Word2Vec(tok)
            
        return torch.Tensor(np.array(output)), torch.Tensor(np.array(alphas))


            



