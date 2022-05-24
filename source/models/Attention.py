import numpy as np
from torch import nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, pretrained, pretrained_dim, embedding_dim, unfreeze_layer_count=None):
        super(Encoder, self).__init__()
        self.preNet = pretrained
        # Turn off gradient for pretrained model
        for param in self.preNet.parameters():
            param.requires_grad = False

        # Exclude top layer
        pre_modules = list(self.preNet.children())[:-2]
        # Unfreeze Last 3 Layer
        # Turn on gradient for unfreeze last 3 layers
        if unfreeze_layer_count:
            for layer in pre_modules[0][-unfreeze_layer_count:]:
                for param_name, param in layer.named_parameters():
                    param.requires_grad = True
                    print(param_name, "requires_grad: {}".format(param.requires_grad))

        self.Net = nn.Sequential(*pre_modules)
        self.dense = nn.Linear(pretrained_dim, embedding_dim)
        #self.pooling = nn.AdaptiveAvgPool2d(latent_pix_dim)

    def forward(self, x):
        x = self.Net(x) # output from pretrained (Batch_size, channel, pix, pix) --> In this case (Batch, 1408, 8, 8)
        #x = x.reshape((x.shape[0], x.shape[1], -1)) # (Batch_size, channel, pix*pix)
        x = torch.moveaxis(x, 1, -1) # (Batch_size, pix, pix, channel)
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
        context_vector = torch.sum(alpha*encoder_feat, dim=1) # (batch, encode_channel)
        return context_vector, alpha.squeeze()

class PyramidAtt(nn.Module): # Feature Pyramid with Bahdanau attention
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(PyramidAtt, self).__init__()
        self.relu = nn.ReLU()
        # Pooling phase
        self.pool_level2 = nn.AvgPool2d(2, 1) # lead to (7,7,D)
        self.pool_level3 = nn.AvgPool2d(4, 1) # lead to (5,5,D)
        # Dense phase
        self.dense_level1 = nn.Linear(encoder_dim, 512)
        self.dense_level2 = nn.Linear(encoder_dim, 512)
        self.dense_level3 = nn.Linear(encoder_dim, 512)
        # Attention phase
        self.encoder_att = nn.Linear(512, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_feat, decoder):
        encoder_feat = encoder_feat.permute(0,3,1,2)
        default_pix = encoder_feat.shape[-1]
        # Pooling
        x_l1 = encoder_feat
        x_l2 = self.pool_level2(encoder_feat)
        x_l3 = self.pool_level3(encoder_feat)
        # Reshape
        x_l1 = encoder_feat = encoder_feat.reshape((encoder_feat.shape[0], encoder_feat.shape[1], -1)).permute(0,2,1)
        x_l2 = x_l2.reshape((x_l2.shape[0], x_l2.shape[1], -1)).permute(0,2,1)
        x_l3 = x_l3.reshape((x_l3.shape[0], x_l3.shape[1], -1)).permute(0,2,1)
        # Dense
        #x_l1 = self.dense_level1(self.relu(x_l1))
        #x_l2 = self.dense_level2(self.relu(x_l2))
        #x_l3 = self.dense_level3(self.relu(x_l3))
        # Encoder attention should be performed on channel dim, not pixel itself.
        # (Batch_size, pixel, embedding_dim)
        # Attention for image latent space
        att1_lv1 = self.encoder_att(x_l1)
        att1_lv2 = self.encoder_att(x_l2)
        att1_lv3 = self.encoder_att(x_l3)
        # Attention for decoder hidden state
        att2 = self.decoder_att(decoder)
        # Calculate attention
        fatt_lv1 = self.tanh(att1_lv1+att2.unsqueeze(1)) # Additive attention (Bahdanau) for pyramid level 1 (Original space)
        fatt_lv2 = self.tanh(att1_lv2+att2.unsqueeze(1)) # Additive attention (Bahdanau) for pyramid level 2
        fatt_lv3 = self.tanh(att1_lv3+att2.unsqueeze(1)) # Additive attention (Bahdanau) for pyramid level 3

        cal_att1 = self.full_att(fatt_lv1) # (batch, pixel, 1)
        cal_att2 = self.full_att(fatt_lv2) # (batch, pixel, 1)
        cal_att3 = self.full_att(fatt_lv3) # (batch, pixel, 1)
    
        # Gathering Bottom-up feature
        cal_att3 = cal_att3.reshape(cal_att3.shape[0], 5, 5).unsqueeze(1)
        cal_att3 = nn.functional.interpolate(cal_att3, size=(default_pix, default_pix), mode="bilinear").reshape((cal_att3.shape[0], -1, 1))

        cal_att2 = cal_att2.reshape(cal_att2.shape[0], 7, 7).unsqueeze(1)
        cal_att2 = nn.functional.interpolate(cal_att2, size=(default_pix, default_pix), mode="bilinear").reshape((cal_att3.shape[0], -1, 1))

        att = cal_att3 + cal_att2 + cal_att1

        alpha = self.softmax(att)
        
        context_vector = torch.sum(alpha*encoder_feat, dim=1) # (batch, encode_channel)
        return context_vector, alpha.squeeze()
        
class AttentionDecoder(nn.Module):
    def __init__(self, encoder_dim, attention_dim, word_emb_size, decoder_hidden, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.word_emb_size = word_emb_size
        self.hidden_size = decoder_hidden
        self.vocab_size = vocab_size
        #self.num_layers = num_layers

        self.attention = PyramidAtt(encoder_dim, decoder_hidden, attention_dim)  # attention network
        self.Word2Vec = nn.Embedding(self.vocab_size, self.word_emb_size)
        #self.gru = nn.GRU(self.word_emb_size, decoder_hidden, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.GRUCell = nn.GRUCell(word_emb_size+encoder_dim, decoder_hidden)
        self.linear = nn.Linear(decoder_hidden, self.vocab_size)
        # Gate context
        self.fc_beta = nn.Linear(decoder_hidden, encoder_dim)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size), device='cuda')

    def forward(self, img, cap, caplen):
        batch_size = img.shape[0]
        num_pixel = img.shape[1] * img.shape[2]
        word_emb = self.Word2Vec(cap)
        h = self.init_hidden(batch_size)
        # Put img latent on top of caption sequence.
        decoder_length = (caplen).tolist() # To find what is the maximum length in this batch.
        predictions = torch.zeros(batch_size, max(decoder_length), self.vocab_size).cuda()
        alphas = torch.zeros((batch_size, max(decoder_length), num_pixel)).cuda()
        # Should be implemented GRUcell manually
        for i in range(1, max(decoder_length)):
            # Calculate Attention scores.
            context_vector, alpha = self.attention(img[:], h)
            # Gate context forward
            gate = self.sigmoid(self.fc_beta(h))
            weighted_context_vector = gate*context_vector
            # Teacher forcing goes here.
            h = self.GRUCell(torch.cat((word_emb[:, i-1, :], weighted_context_vector), dim=1), h[:])
            preds = self.linear(h) # (batch, size_vocab)
            predictions[:, i, :] = preds
            alphas[:, i, :] = alpha

        predictions[:, 0, 1] = 1 # Mark at <start> token
        return predictions, decoder_length, alphas

    def inference(self, img, max_len=30):
        batch_size = img.shape[0]
        num_pixel = img.shape[1] * img.shape[2]
        output = np.ones((batch_size, max_len))
        alphas = np.zeros((batch_size, max_len, num_pixel))
        inputs = self.Word2Vec(torch.ones((batch_size), dtype=torch.long).cuda())
        h = self.init_hidden(batch_size)
        step = 1
        while True:
            context_vector, alpha = self.attention(img[:], h[:])
            # Gate context forward
            gate = self.sigmoid(self.fc_beta(h))
            weighted_context_vector = gate*context_vector
            # GRU step
            h = self.GRUCell(torch.cat((inputs, weighted_context_vector), dim=1), h[:])
            out = self.linear(h)
            out = out.squeeze(1)
            tok = torch.argmax(out, dim=1)

            output[:, step] = tok.cpu().numpy()
            alphas[:, step, :] = alpha.cpu().numpy()
            #output.append(tok.cpu().numpy()[0].item())
            #alphas.append(alpha.cpu().numpy())
            
            if step>=(max_len-1):
                # We found <end> token, Stop prediction
                break

            step += 1
            inputs = self.Word2Vec(tok)
            
        return output, alphas


            



