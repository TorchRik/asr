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
        res = torch.bmm(sm, h).squeeze()
        return res.reshape((-1, res.shape[-1]))


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
        self.hidden_state: tuple[torch.Tensor | None, torch.Tensor | None] = (
            None,
            None,
        )

    def init_state(self, bs: int) -> None:
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros((self.num_layers, bs, self.hidden_dim), device=device),
            torch.zeros((self.num_layers, bs, self.hidden_dim), device=device),
        )

    def get_hidden_state_for_att(self) -> torch.Tensor:
        """Return state of all layers as query for attention"""
        return self.hidden_state[0][(0,), :, :].transpose(0, 1)

    def get_hidden_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return state of all layers as query for attention"""
        return self.hidden_state

    def set_hidden_state(
        self, new_hidden_state: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Return state of all layers as query for attention"""
        self.hidden_state = new_hidden_state

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
            torch.zeros(bs, dtype=torch.long, device=encode_feature.device)
        )
        all_log_probs = []
        attention_contexts = []
        idxs = []
        for t in range(decode_step):
            context = self.attention(
                h=encode_feature,
                s=self.decoder.get_hidden_state_for_att(),
            )
            attention_contexts.append(context)
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
                    sampled_char_idx = (
                        torch.distributions.Categorical(cur_prob).sample().squeeze()
                    )
            embed_predict = self.emb(sampled_char_idx.squeeze())
            idxs.append(sampled_char_idx)
            all_log_probs.append(log_probs.squeeze())
        idxs = torch.stack(idxs, dim=1).detach().cpu()
        length = torch.argmax((idxs == 0).int(), dim=1)
        length[length == 0] = idxs.shape[1]
        return {
            "predicted_idxs": idxs,
            "log_probs": torch.stack(all_log_probs, dim=1),
            "log_probs_length": length,
            "attention_contexts": torch.stack(attention_contexts, dim=1).detach().cpu(),
        }

    def beam_search(
        self,
        spectrogram: torch.Tensor,
        max_length: int = 200,
        beam_size: int = 5,
        *args,
        **kwargs
    ):
        spectrogram = spectrogram.permute(0, 2, 1)
        bs = spectrogram.shape[0]

        # I have no idea how it can work for batch, so it will be single version
        assert bs == 1
        encode_feature = self.encoder(spectrogram)
        embed_predict = self.emb(
            torch.zeros(1, dtype=torch.long, device=spectrogram.device)
        )
        self.decoder.init_state(1)
        hidden_state = self.decoder.get_hidden_state()

        beams = [(embed_predict, hidden_state, 0.0, False, [])]

        for _ in range(max_length):
            new_beams = []

            for (
                embed_predict,
                hidden_state,
                current_score,
                is_end,
                current_seq,
            ) in beams:
                if is_end:
                    new_beams.append(
                        (
                            embed_predict,
                            hidden_state,
                            current_score,
                            is_end,
                            current_seq,
                        )
                    )
                    continue

                self.decoder.set_hidden_state(hidden_state)
                context = self.attention(
                    h=encode_feature,
                    s=self.decoder.get_hidden_state_for_att(),
                )
                decoder_out = self.decoder(
                    input_token_embed=embed_predict, attention_res=context
                )
                latents = self.char_trans(decoder_out)
                log_probs = F.log_softmax(latents, dim=-1).squeeze()

                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    next_token = topk_indices[i].item()

                    next_score = current_score + topk_log_probs[i].item()

                    next_seq = current_seq + [next_token]

                    right_shape = embed_predict.shape
                    embed_predict = self.emb(topk_indices[i]).reshape(right_shape)

                    new_beams.append(
                        (
                            embed_predict,
                            self.decoder.get_hidden_state(),
                            next_score,
                            next_token == 0,
                            next_seq,
                        )
                    )
            beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

        return {
            "predicted_idxs": torch.tensor([beams[0][-1]]),
            "log_probs_length": torch.tensor([len(beams[0][-1])]),
        }
