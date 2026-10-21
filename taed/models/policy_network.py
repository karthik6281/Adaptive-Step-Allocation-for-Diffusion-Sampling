import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_dim=64,
        feature_dim=32,
        num_classes=None,
        class_embedding_dim=16,
        max_diffusion_steps=1000,
    ):
        """
        LSTM-based policy network. Can optionally handle class conditioning.

        Args:
            max_diffusion_steps: Maximum timesteps from diffusion model scheduler
        """
        super().__init__()
        if min(input_channels, hidden_dim, feature_dim, max_diffusion_steps) < 1:
            raise ValueError("Policy dimensions and max_diffusion_steps must be positive.")
        self.is_conditional = num_classes is not None
        self.max_diffusion_steps = max_diffusion_steps
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.time_embed = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        time_embed_dim = 16
        self.class_embed = None
        current_lstm_input_size = feature_dim + time_embed_dim
        if self.is_conditional:
            if num_classes is None:
                raise ValueError("num_classes must be provided for conditional network")
            if not isinstance(class_embedding_dim, int) or class_embedding_dim <= 0:
                raise ValueError(
                    "class_embedding_dim must be a positive integer for conditional network"
                )
            self.class_embed = nn.Embedding(num_classes, class_embedding_dim)
            current_lstm_input_size += class_embedding_dim
        self.lstm = nn.LSTM(
            input_size=current_lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.skip_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, x_t, t, hidden=None, class_labels=None):
        if not self.is_conditional and class_labels is not None:
            raise ValueError("This policy does not accept class labels.")
        batch_size = x_t.shape[0]
        device = x_t.device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        image_features = self.feature_extractor(x_t).view(batch_size, -1)
        t_norm = t.float().unsqueeze(-1) / float(self.max_diffusion_steps)
        time_embedding = self.time_embed(t_norm)
        features_list = [image_features, time_embedding]
        if self.is_conditional:
            if class_labels is None:
                raise ValueError("class_labels must be provided for conditional forward")
            if self.class_embed is None:
                raise RuntimeError("Conditional but class_embed is None")
            if class_labels.shape[0] != batch_size:
                raise ValueError("Batch size mismatch: x_t vs class_labels")
            if class_labels.dtype != torch.long:
                raise ValueError("Class labels must have dtype torch.long.")
            class_embedding = self.class_embed(class_labels.to(device))
            features_list.append(class_embedding)
        combined_features = torch.cat(features_list, dim=-1)
        lstm_input = combined_features.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)
        lstm_output = lstm_out.squeeze(1)
        skip_logit = self.skip_head(lstm_output)
        skip_prob = torch.sigmoid(skip_logit).view(batch_size)
        value = self.value_head(lstm_output).view(batch_size)
        correction = None
        return (skip_prob, new_hidden, value, correction)
