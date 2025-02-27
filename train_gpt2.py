from dataclasses import dataclass

import torch
import torch.nn as nn
import tiktoken
import time
import math 
import inspect
import torch.distributed as dist
import os

from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"using device: {device}")
    return device

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0 
        
        # key query and value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        #mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, seq len, n_embd
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, seq_len, head_size
        
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_head, seq_len, seq_len) x (B, n_head, seq_len, head_size) -> (B, n_head, seq_len, head_size)
        
        # lines above are replaced to the flash attention optimized version
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all heads outputs
        y = self.c_proj(y)
        
        return y
        
        
    
class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
        
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        #init params
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
    @classmethod
    def from_pretrained(clas, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        
        print("loading weights from pretrained gpt: %s", model_type)
        
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #ignore mask
        
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type, cache_dir='./cache')
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay=0.01, lr=6e-4, device='cpu'):
        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        
        print('weight decay params: ', num_decay_params)
        print('weight no-decay params: ', num_no_decay_params)
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        
        
        
    
class DataLoaderLite:
    def __init__(self, B, T, local_rank=0, world_size=1):
        self.B = B
        self.T = T
        self.local_rank = local_rank
        self.world_size = world_size
        
        with open('input.txt') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        
        print(f"total tokens: {len(self.tokens)}")
        print(f"total batches: {len(self.tokens) // (B*T)}")
        
        self.current_pos = local_rank * B * T
    
    def __iter__(self):
        return self
    
    def __next__(self):
        B, T, local_rank, world_size = self.B, self.T, self.local_rank, self.world_size
        if self.current_pos + B * T + 1 > len(self.tokens):
            self.current_pos = local_rank * B * T
        
        buf = self.tokens[self.current_pos:self.current_pos + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_pos += B * T * world_size
        return x, y

#-------------Training loop----------------
torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

if hasattr(torch, 'mps'):
    torch.mps.manual_seed(1337)
    
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    device = get_device()
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    torch.cuda.set_device(device)
    master_process = True
    
print(f"STARTING | ddp_rank: {ddp_rank} | ddp_local_rank: {ddp_local_rank} | ddp_world_size: {ddp_world_size} | device: {device}")

total_batch_size = 524288 #batch size
B = 16 #micro batch size
T = 1024 #seq len

assert total_batch_size % (B * T * ddp_world_size) == 0 , "total batch size should be multiple of microbatch size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Batch size: {total_batch_size} | micro batch size: {B} | seq len: {T} | Grad accumulate steps: {grad_accum_steps}")

# data loader
train_loader = DataLoaderLite(B=16, T=1024, local_rank=ddp_local_rank, world_size=ddp_world_size)

# train the model
model = GPT(GPTConfig(vocab_size=50304)).to(device)
if device=='mps':
    model = torch.compile(model, backend="aot_eager")
else:
    model = torch.compile(model)
    
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps = 50
weight_decay = 0.1

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(decay_ratio * math.pi))
    return min_lr + coeff * (max_lr - min_lr)
    


#optimzier
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(lr=max_lr, weight_decay=weight_decay, device=device)

for step in range(max_steps):
    loss_accum = 0 
    start_time = time.time()

    for micro_step in range(grad_accum_steps):
        batch, target = next(train_loader)
        batch, target = batch.to(device), target.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(batch, target)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, po=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
        
    dt = (time.time() - start_time) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (dt/1000)
    if master_process:
        print(f"step {step} | loss: {loss_accum.item():.4f} | time: {dt:.2f} | tokens/sec: {tokens_per_sec:.2f} | norm: {norm:.4f} | lr: {lr}")

if ddp:
    destroy_process_group()

# import tiktoken
# enc = tiktoken.get_encoding('gpt2')

# with open('input.txt') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)

# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)
# 
# from time import time
# start = time()
# for i in range(50):
#     
#     
#     loss.backward()
#     optimizer.step()
    
#     print(f"step {i}, loss: {loss.item()}")
# print(f"training took {time() - start} seconds")

# tokens = enc.encode("Hello, I'm language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
# x = tokens.to(device)

# torch.manual_seed(10)
# torch.mps.manual_seed(10)

# while x.size(1) < max_length:
#     with torch.no_grad():
#         logits = model(x)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=1)
#         topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)
#         ix = torch.multinomial(topk_probs, 1)
#         xcol = torch.gather(topk_indicies, -1, ix)
#         x = torch.cat((x, xcol), dim=1)
        
# for i in range(num_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens=tokens)
#     print(f"> {decoded}")