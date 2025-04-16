import torch
import torch.nn as nn
from transformers import AutoModel


class Wav2VecClassifier(nn.Module):
    """
    A classification model using a pre-trained Wav2Vec 2.0 model.

    Args:
        model_name (str): Name of the pre-trained Wav2Vec model
                          (e.g., "facebook/wav2vec2-base-960h").
        num_classes (int): Number of output classes for the classifier.
        freeze_feature_extractor (bool): If True, freeze the parameters of the
                                         pre-trained Wav2Vec model. Only the
                                         classifier head will be trained.
    """
    def __init__(self, model_name="facebook/wav2vec2-base-960h", num_classes=1, freeze_feature_extractor=True):
        super().__init__()

        # Load the pre-trained Wav2Vec model
        # AutoModel automatically selects the correct architecture
        self.wav2vec = AutoModel.from_pretrained(model_name)

        # Get the hidden size of the Wav2Vec model's output
        # This is needed for the input size of the classifier
        hidden_size = self.wav2vec.config.hidden_size

        # Define the classification head
        # We use a simple linear layer on top of the pooled Wav2Vec output
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Freeze the Wav2Vec model layers if requested
        if freeze_feature_extractor:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        # Ensure the classifier parameters are trainable (they are by default)
        # for param in self.classifier.parameters():
        #      param.requires_grad = True # Default behavior

        print(f"Loaded Wav2Vec model: {model_name}")
        print(f"Feature extractor frozen: {freeze_feature_extractor}")
        print(f"Output hidden size: {hidden_size}")
        print(f"Number of classes: {num_classes}")


    def forward(self, input_values, attention_mask=None):
        """
        Forward pass of the model.

        Args:
            input_values (torch.Tensor): Processed waveform tensor.
                 Shape: [Batch, SequenceLength]
                 Obtained from the Wav2Vec2Processor/AutoProcessor.
            attention_mask (torch.Tensor, optional): Mask to avoid performing
                 attention on padding token indices.
                 Shape: [Batch, SequenceLength]
                 Also obtained from the Wav2Vec2Processor/AutoProcessor.

        Returns:
            torch.Tensor: Raw logits from the classifier. Shape: [Batch, num_classes]
        """
        # Pass the processed waveforms through the Wav2Vec model
        # The output contains fields like 'last_hidden_state', 'extract_features', etc.
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)

        # We use the 'last_hidden_state' which has shape [Batch, SequenceLength, HiddenSize]
        hidden_states = outputs.last_hidden_state

        # Pool the hidden states across the time dimension (SequenceLength)
        # Mean pooling is a common strategy
        # Output shape: [Batch, HiddenSize]
        pooled_output = torch.mean(hidden_states, dim=1)

        # Pass the pooled output through the classifier
        # Output shape: [Batch, num_classes]
        logits = self.classifier(pooled_output)

        return logits
