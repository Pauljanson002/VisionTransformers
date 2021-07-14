import torch.nn as nn
from layers import Tokenizer, Transformer


class DeiTLite(nn.Module):
    def __init__(self,
                 img_size=32,
                 embedding_dim=256,
                 n_input_channels=3,
                 patch_size=4,
                 *args, **kwargs
                 ):
        super(DeiTLite,self).__init__()
        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            max_pool=False,
            activation=None,
            n_conv_layers=1,
            conv_bias=True
        )
        self.classifier = Transformer(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout_rate=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            distill=True,
            *args, **kwargs
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)
