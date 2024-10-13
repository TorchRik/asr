import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 80,
        conv_dim: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        hidden_dim: int = 1024,
        lstm_layers: int = 4,
    ):
        self.in_channel, self.freq_dim, self.out_cnn_dim = input_dim // 40, 40, 320

        super(Encoder, self).__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
        )
        # TODO add batch normalization between LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.out_cnn_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, ts, _ = x.shape
        x = x.view(bs, ts, self.in_channel, self.freq_dim).transpose(1, 2)
        x_cnn = self.cnn_block(x).transpose(1, 2)
        output, _ = self.lstm(x_cnn.reshape(x_cnn.shape[0], x_cnn.shape[1], -1))
        return output


class Attention(nn.Module):
    def __init__(self, h_size: int = 1024, w_size: int = 1024):
        super(Attention, self).__init__()
        self.linear_for_h = nn.Linear(in_features=h_size, out_features=h_size)
        self.linear_for_w = nn.Linear(in_features=w_size, out_features=h_size)

    def forward(self, h, s):
        h = self.linear_for_h(h)
        s = self.linear_for_w(s)
        attention_scores = torch.bmm(h, s.transpose(1, 2))
        sm = torch.softmax(attention_scores.view(-1, h.size(1)), dim=1).view(
            h.size(0), -1, h.size(1)
        )
        return torch.bmm(sm, h).squeeze()


class Decoder(nn.Module):
    def __init__(
        self, hidden_dim: int, num_layers: int, embedding_size: int, encoder_size: int
    ):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=encoder_size + embedding_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hidden_state = (None, None)

    def init_state(self, bs: int) -> None:
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros((self.num_layers, bs, self.hidden_dim), device=device),
            torch.zeros((self.num_layers, bs, self.hidden_dim), device=device),
        )

    def get_hidden_state_for_att(self):
        """Return state of all layers as query for attention"""
        return self.hidden_state[0][(0,), :, :].transpose(0, 1)

    def forward(
        self,
        input_token_embed: torch.Tensor,
        attention_res: torch.Tensor,
    ):
        lstm_input = torch.cat((input_token_embed, attention_res), dim=-1)
        lstm_input = lstm_input.reshape(lstm_input.size(0), -1, lstm_input.size(1))

        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)

        return lstm_out


class LAS(nn.Module):
    def __init__(
        self,
        spec_dim: int,
        listener_lstm_layers: int,
        listener_hidden_dim: int,
        speller_hidden_dim: int,
        speller_lstm_layers: int,
        embedding_size: int,
        vocab_size: int,
        *args,
        **kwargs
    ):
        super(LAS, self).__init__()
        self.encoder = Encoder(
            input_dim=spec_dim,
            hidden_dim=listener_hidden_dim,
            lstm_layers=listener_lstm_layers,
        )
        self.attention = Attention(
            h_size=listener_hidden_dim * 2, w_size=speller_hidden_dim
        )

        self.decoder = Decoder(
            hidden_dim=speller_hidden_dim,
            embedding_size=embedding_size,
            num_layers=speller_lstm_layers,
            encoder_size=listener_hidden_dim * 2,
        )
        self.char_trans = nn.Linear(speller_hidden_dim, vocab_size)
        self.emb = nn.Embedding(vocab_size, embedding_size)

    def forward(
        self,
        spectrogram: torch.Tensor,
        text_encoded: torch.Tensor,
        text_encoded_lengths: torch.Tensor,
        is_train: bool,
        tf_rate: float = 0.9,
        *args,
        **kwargs
    ):
        spectrogram = spectrogram.permute(0, 2, 1)
        if not is_train:
            tf_rate = 0
        decode_step = text_encoded_lengths.max()
        bs = spectrogram.shape[0]
        encode_feature = self.encoder(spectrogram)

        self.decoder.init_state(bs)

        embed_predict = self.emb(
            torch.zeros((bs), dtype=torch.long, device=encode_feature.device)
        )
        all_log_probs = []
        idxs = []
        for t in range(decode_step):
            context = self.attention(
                h=encode_feature,
                s=self.decoder.get_hidden_state_for_att(),
            )
            decoder_out = self.decoder(
                input_token_embed=embed_predict, attention_res=context
            )
            latents = self.char_trans(decoder_out)
            log_probs = F.log_softmax(latents, dim=-1)
            if text_encoded is not None and torch.rand(1).item() <= tf_rate:
                sampled_char_idx = text_encoded[:, t]
            else:
                with torch.no_grad():
                    cur_prob = latents.softmax(dim=-1)
                    sampled_char_idx = torch.distributions.Categorical(
                        cur_prob
                    ).sample()
            embed_predict = self.emb(sampled_char_idx.squeeze())
            idxs.append(sampled_char_idx)
            all_log_probs.append(log_probs.squeeze())
        return {
            "predicted_idxs": idxs,
            "log_probs": torch.stack(all_log_probs, dim=1),
            "log_probs_length": text_encoded_lengths,
        }
