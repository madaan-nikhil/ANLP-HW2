from torch import nn
import torch

class TokenClassification(nn.Module):
    
    def __init__(self, 
                feature_extractor,
                device,
                num_labels,
                hidden_size,
                dropout):
        super(TokenClassification, self).__init__()
        self.feature_extractor = feature_extractor
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.device = device
    
    def forward(self, input):
        
        outputs = self.distilbert(
            **input
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return logits