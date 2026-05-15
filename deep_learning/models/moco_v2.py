import torch
from torch import nn
import torch.nn.functional as functional
from torchvision import models


class MocoProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature):
        return self.layers(feature)


def build_resnet18_encoder(feature_dim):
    encoder = models.resnet18(weights=None)
    input_dim = encoder.fc.in_features
    encoder.fc = MocoProjectionHead(input_dim, input_dim, feature_dim)
    return encoder


class MocoV2(nn.Module):
    def __init__(self, feature_dim=128, queue_size=1024, momentum=0.999, temperature=0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.encoder_q = build_resnet18_encoder(feature_dim)
        self.encoder_k = build_resnet18_encoder(feature_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer('queue', functional.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size >= self.queue_size:
            self.queue = keys[-self.queue_size:].T
            self.queue_ptr[0] = 0
            return
        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[:, ptr:end_ptr] = keys.T
        else:
            first_len = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_len].T
            self.queue[:, :end_ptr - self.queue_size] = keys[first_len:].T
        self.queue_ptr[0] = end_ptr % self.queue_size

    def forward(self, image_q, image_k):
        query = functional.normalize(self.encoder_q(image_q), dim=1)
        with torch.no_grad():
            self.momentum_update_key_encoder()
            key = functional.normalize(self.encoder_k(image_k), dim=1)

        positive_logits = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        negative_logits = torch.einsum('nc,ck->nk', [query, self.queue.clone().detach()])
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self.dequeue_and_enqueue(key)
        return logits, labels


class FrozenClipMocoV2(nn.Module):
    def __init__(self, clip_model, clip_dim=512, hidden_dim=1024, feature_dim=128, queue_size=2048, momentum=0.999, temperature=0.2):
        super().__init__()
        self.clip_model = clip_model
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.projector_q = MocoProjectionHead(clip_dim, hidden_dim, feature_dim)
        self.projector_k = MocoProjectionHead(clip_dim, hidden_dim, feature_dim)

        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer('queue', functional.normalize(torch.randn(feature_dim, queue_size), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def trainable_parameters(self):
        return self.projector_q.parameters()

    @torch.no_grad()
    def encode_clip_image(self, image):
        feature = self.clip_model.encode_image(image)
        return feature.float()

    @torch.no_grad()
    def momentum_update_key_projector(self):
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size >= self.queue_size:
            self.queue = keys[-self.queue_size:].T
            self.queue_ptr[0] = 0
            return
        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[:, ptr:end_ptr] = keys.T
        else:
            first_len = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_len].T
            self.queue[:, :end_ptr - self.queue_size] = keys[first_len:].T
        self.queue_ptr[0] = end_ptr % self.queue_size

    def forward(self, image_q, image_k):
        with torch.no_grad():
            clip_q = self.encode_clip_image(image_q)
            clip_k = self.encode_clip_image(image_k)
        query = functional.normalize(self.projector_q(clip_q), dim=1)
        with torch.no_grad():
            self.momentum_update_key_projector()
            key = functional.normalize(self.projector_k(clip_k), dim=1)

        positive_logits = torch.einsum('nc,nc->n', [query, key]).unsqueeze(-1)
        negative_logits = torch.einsum('nc,ck->nk', [query, self.queue.clone().detach()])
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self.dequeue_and_enqueue(key)
        return logits, labels

    def checkpoint_state(self):
        return {
            'projector_q': self.projector_q.state_dict(),
            'projector_k': self.projector_k.state_dict(),
            'queue': self.queue,
            'queue_ptr': self.queue_ptr,
            'feature_dim': self.feature_dim,
            'queue_size': self.queue_size,
            'momentum': self.momentum,
            'temperature': self.temperature,
        }


def save_moco_checkpoint(path, model, optimizer, epoch, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model': model.checkpoint_state() if hasattr(model, 'checkpoint_state') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }
    temp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)
