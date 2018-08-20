import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.0,\
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        features = features.view(len(features), 1, -1)
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features, embeddings), 1)
        
        out, hidden = self.lstm(inputs)
        out = self.linear(out)
        
        return out

    def sample(self, features, states=None, max_len=20):
        for i in range(max_len):
            
            if states is None:
                inputs = features
            else:
                embeddings = self.embed(states)
                inputs = torch.cat((features, embeddings), 1)
            out, hidden = self.lstm(inputs)
            out = self.linear(out)
            
            val, states = out.max(2)

        outputs = states.tolist()[0]
        return outputs
    
  