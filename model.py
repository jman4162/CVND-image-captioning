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
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob = 0.3):
        
        """
        Set the hyper-parameters and build the layers.
        Parameters
        ----------
        - embed_size  : Dimensionality of image and word embeddings
        - hidden_size : number of features in hidden state of the RNN decoder
        - vocab_size  : The size of vocabulary or output size
        - num_layers  : Number of layers
        """
        
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded vectors as inputs and outputs hidden states of hidden_size
        self.lstm = nn.LSTM(input_size  = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers  = num_layers,
                            dropout     = drop_prob,
                            batch_first = True)
        
        # define dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # the linear layer that maps the hidden state output dimension 
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """Extract the image feature vectors."""

        # Remove end token from captions
        captions = captions[:,:-1] 
        
        # Get work embeddings from image captions
        embeddings = self.word_embeddings(captions)
        
        # Concatenating features to embedding
        # torch.cat 3D tensors
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # LSTM layer
        lstm_out, hidden = self.lstm(inputs)
        
        # pass lstm_out through a droupout layer
        lstm_out = self.dropout(lstm_out)
        
        outputs = self.fc(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Initialize caption
        caption = []
        
        # Loop through list of tensor ids of length max_len
        for i in range(max_len):
            # Pass through LSTM layer
            lstm_out, states = self.lstm(inputs, states)
            
            # Pass through linear layer
            outputs = self.fc(lstm_out)
            
            # Squeeze output
            outputs = outputs.squeeze(1)
            
            # Select maximum probility to choose word
            target = outputs.max(1)[1]
            
            # Append result to caption list
            caption.append(target.item())
            
            # Break the sequence early if we see the <end> word
            if (target == 1):
                break
            
            # Prepare network for next loop, update the input for next iteration
            # Embed last predicted word to be the new input of the LSTM
            inputs = self.word_embeddings(target).unsqueeze(1)
            
        return caption