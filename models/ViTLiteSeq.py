from models.ViTLite import ViTLite
from layers import Transformer


class ViTLiteSeq(ViTLite):
    def __init__(self,
                 img_size=32,
                 embedding_dim=256,
                 n_input_channels=3,
                 patch_size=4,
                 *args, **kwargs
                 ):
        super(ViTLiteSeq, self).__init__(
            img_size=img_size,
            embedding_dim=embedding_dim,
            n_input_channels=n_input_channels,
            patch_size=patch_size,
            *args, **kwargs
        )
        self.classifier = Transformer(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.1,
            attention_dropout=0.,
            stochastic_depth=0.,
            *args, **kwargs)
