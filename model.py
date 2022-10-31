from torch import nn
import torch

class TokenClassification(nn.Module):
    
    def __init__(self, 
                feature_extractor,
                device,
                num_labels,
                hidden_size,
                dropout,
                alternate_objective=False):
        super(TokenClassification, self).__init__()
        self.feature_extractor = feature_extractor
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.classifier_alternate_objective = nn.Linear(hidden_size, 8)
        self.device = device
    
    def forward(self, input, alternate_objective=False):
        
        outputs = self.feature_extractor(
            **input
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if alternate_objective:
          logits = self.classifier_alternate_objective(sequence_output)
          return logits

        logits = self.classifier(sequence_output)

        return logits