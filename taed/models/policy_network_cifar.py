import torch
import torch.nn as nn


class PolicyNetworkCIFAR(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_dim=64,
        feature_dim=32,
        num_classes=None,
        class_embedding_dim=16,
        max_diffusion_steps=1000,
    ):
        """Compact LSTM-based policy network for CIFAR-10 (unconditional)."""
        super().__init__()
        if num_classes is not None:
            raise ValueError(
                "PolicyNetworkCIFAR is unconditional; use PolicyNetwork for class labels."
            )
        if min(input_channels, hidden_dim, feature_dim, max_diffusion_steps) < 1:
            raise ValueError("Policy dimensions and max_diffusion_steps must be positive.")
        self.is_conditional = False
        self.max_diffusion_steps = max_diffusion_steps
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 72, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.Conv2d(72, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.time_embed = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 16))
        time_embed_dim = 16
        self.class_embed = None
        current_lstm_input_size = feature_dim + time_embed_dim
        self.lstm = nn.LSTM(
            input_size=current_lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.skip_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def forward(self, x_t, t, hidden=None, class_labels=None):
        if class_labels is not None:
            raise ValueError("PolicyNetworkCIFAR does not accept class labels.")
        batch_size = x_t.shape[0]
        device = x_t.device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        image_features = self.feature_extractor(x_t).view(batch_size, -1)
        t_norm = t.float().unsqueeze(-1) / float(self.max_diffusion_steps)
        time_embedding = self.time_embed(t_norm)
        features_list = [image_features, time_embedding]
        combined_features = torch.cat(features_list, dim=-1)
        lstm_input = combined_features.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)
        lstm_output = lstm_out.squeeze(1)
        skip_logit = self.skip_head(lstm_output)
        skip_prob = torch.sigmoid(skip_logit).view(batch_size)
        value = self.value_head(lstm_output).view(batch_size)
        correction = None
        return (skip_prob, new_hidden, value, correction)
