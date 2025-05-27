import torch.nn as nn
import torch
from transformers import BertConfig, BertModel
from .base import Model
from rfml.nn.layers import Flatten, PowerNormalization


class transformerBERT(Model):
    """Custom BERT-style Transformer for RF classification.

    This model adapts the BERT architecture for non-text (e.g., I/Q RF) inputs.
    It is initialized from scratch and trained end-to-end.

    Architecture is based on the encoder portion of the original BERT model,
    configured with reduced layers and hidden size for RF compatibility.
    """

    def __init__(self, input_samples: int, num_classes: int):
        super().__init__(input_samples, num_classes)
        self.preprocess = PowerNormalization()
        config = BertConfig(
            vocab_size=1,  # Not used, but required
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=input_samples,
            type_vocab_size=1,
        )

        self.bert = BertModel(config)

        # Project (I/Q input) → hidden_size
        self.input_proj = nn.Linear(2, config.hidden_size)  # 2 channels: I and Q
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Linear(config.hidden_size, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.preprocess(x)
        x = x.squeeze(1)             # → (batch_size, 2, 128)
        x = x.permute(0, 2, 1)       # → (batch_size, seq_len=128, channels=2)
        x = self.input_proj(x)       # → (batch_size, seq_len, hidden_size)

        x = self.norm(x) 

        # Create dummy attention mask
        attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)

        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        
        logits = self.classifier(self.dropout(cls_output))
        return logits

    # def _freeze(self):
    #     """Freeze BERT transformer layers but allow classifier to train."""
    #     for name, param in self.bert.named_parameters():
    #         param.requires_grad = False

    def _unfreeze(self):
        """Enable training of all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
