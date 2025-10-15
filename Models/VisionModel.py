import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
from safetensors.torch import load_file

class VisionModel(nn.Module):
    def __init__(self, image_size=448):
        super().__init__()
        self.image_size = image_size
        
        # Patch embeddings with conv stem structure
        self.patch_embeddings = nn.ModuleDict({
            'conv': nn.ModuleList([
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ModuleDict({'norm': nn.LayerNorm([64, image_size//2, image_size//2])}),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ModuleDict({'norm': nn.LayerNorm([128, image_size//4, image_size//4])}),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ModuleDict({'norm': nn.LayerNorm([256, image_size//8, image_size//8])}),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ModuleDict({'norm': nn.LayerNorm([512, image_size//16, image_size//16])}),
                nn.ReLU(),
                nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0, bias=True),
            ])
        })
        
        # Transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(12):  # 12 transformer blocks
            block = nn.ModuleDict()
            block['norm1'] = nn.LayerNorm(768)
            block['qkv_proj'] = nn.Linear(768, 768 * 3)
            block['out_proj'] = nn.Linear(768, 768)
            block['norm2'] = nn.LayerNorm(768)
            block['mlp'] = nn.ModuleDict({
                'linear1': nn.Linear(768, 768 * 4),
                'linear2': nn.Linear(768 * 4, 768)
            })
            
            # Create wrapper modules for skip connections
            class SkipModule(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.skip = nn.Parameter(torch.zeros(1))
            
            block['skip_init1'] = SkipModule()
            block['skip_init2'] = SkipModule()
            self.blocks.append(block)
            
        # Final layers
        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, 5813)  # 5813 tags in the dataset (from config)
        
    @staticmethod
    def load_model(path):
        path = Path(path)
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
            
        model = VisionModel(image_size=config.get('image_size', 448))
        model_path = path / 'model.safetensors'
        try:
            if model_path.exists():
                # Try loading with safetensors first
                try:
                    state_dict = load_file(model_path)
                except Exception as e:
                    print(f"Failed to load with safetensors: {e}")
                    state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
            else:
                # Try .pt or .pth file as fallback
                for ext in ['.pt', '.pth']:
                    if (path / f'model{ext}').exists():
                        model_path = path / f'model{ext}'
                        break
                print(f"Loading model from {model_path}")
                state_dict = torch.load(model_path, weights_only=False, map_location='cpu')
            
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
        return model

    def forward(self, batch):
        x = batch['image']
        
        # Patch embedding with conv stem
        for layer in self.patch_embeddings['conv']:
            if isinstance(layer, nn.ModuleDict):
                x = layer['norm'](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = layer(x)
        
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention
            h = block['norm1'](x)
            qkv = block['qkv_proj'](h)
            qkv = qkv.reshape(B, -1, 3, 12, 64).permute(2, 0, 3, 1, 4)  # (3, B, 12, N, 64)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) / 8
            attn = F.softmax(attn, dim=-1)
            
            h = (attn @ v).transpose(1, 2).reshape(B, -1, 768)
            h = block['out_proj'](h)
            x = x + h * block['skip_init1'].skip
            
            # MLP
            h = block['norm2'](x)
            h = block['mlp']['linear1'](h)
            h = F.gelu(h)
            h = block['mlp']['linear2'](h)
            x = x + h * block['skip_init2'].skip
        
        # Final layers
        x = self.norm(x)
        x = x.mean(dim=1)  # global pool
        x = self.head(x)
        
        return {'tags': x}
