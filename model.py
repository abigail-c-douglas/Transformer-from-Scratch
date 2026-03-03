import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from jaxtyping import Float

@dataclass
class Config:
    d_model: int 
    d_vocab: int
    d_hidden: int
    d_embedding: int


class Tokenizer:
    def __init__(self, text):
        cleaned = self.clean_text(text)
        self.chars = sorted(set(cleaned))
        self.vocab_size = len(self.chars)
        self.d_vocab = self.vocab_size
        self.encode = {}
        self.decode = {}    
        for i, char in enumerate(self.chars): 
            self.encode[char] = i
            self.decode[i]= char

    def clean_text(self, text : str) -> str:
        text = text.lower()
        cleaned = []
        for char in text: 
            if char.isalpha() or char in " .?!": 
                cleaned.append(char)
        cleaned_text = "".join(cleaned)
        return cleaned_text

        
    def tokenize(self, text):
        cleaned = self.clean_text(text)
        tokens = []
        for char in cleaned: 
            if char in self.encode:
                tokens.append(self.encode[char])    
        return tokens 
    
    def detokenize(self, tokens):
        word_list = []
        for id in tokens: 
            if id in self.decode:
                word_list.append(self.decode[id])
        detokenized = "".join(word_list)
        return detokenized


class MLP(nn.Module):
    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.linear_up = nn.Linear(config.d_model, config.d_hidden)
        self.linear_down = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x:Float[torch.Tensor, "* d_model"]) -> Float [torch.Tensor, "* d_model"]:
        x = self.linear_up(x)
        x = torch.relu(x)
        x = self.linear_down(x)
        return x 
    
class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden
        self.W_q = nn.Linear(self.d_model, self.d_hidden)
        self.W_k = nn.Linear(self.d_model, self.d_hidden)
        self.W_ov = nn.Linear(self.d_model, self.d_hidden)
        self.W_o = nn.Linear(self.d_hidden, self.d_model) 

    def forward(self, x: Float[torch.Tensor, "n_ctx d_model"]) -> Float[torch.Tensor, "n ctx d_model"]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_ov(x)
        n_ctx = x.shape[-2]
        M = torch.triu(
            torch.full((n_ctx, n_ctx), float("-inf"), device=x.device),
            diagonal = 1
        )
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        attn_weights = F.softmax(scores + M, dim = -1)
        output = self.W_o(torch.matmul(attn_weights, V))
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.A = AttentionHead(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Float[torch.Tensor, "n_ctx d_model"]) -> Float[torch.Tensor, "n_ctx d_model"]:
       x = x + self.A(x)
       x = x + self.mlp(x)
       return x
    
class Transformer(nn.Module):
    def __init__(self, config: Config, tokenizer : Tokenizer, max_seq_length, num_blocks):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.token_embedding = nn.Embedding(config.d_vocab, config.d_model) 
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, config.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(config.d_model) 
        self.unembedding = nn.Linear(config.d_model, config.d_vocab)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.token_embedding(x) + self.pos_embedding[:, :seq_length, :]
        for block in self.blocks: 
            x = block(x)
        x = self.ln(x)
        logits = self.unembedding(x)
        return logits
    
    def generate(self, text: str, max_length:int, temperature: float = 0.5 )-> str:
        self.eval()
        tokens = self.tokenizer.tokenize(text)
        tokens = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            for i in range(max_length):
                tokens_short = tokens[:, -self.pos_embedding.shape[1]:]
                logits = self.forward(tokens_short) 
                next_logits = logits[0, -1, :]
                probabilities = F.softmax(next_logits / temperature, dim = -1)
                new_token = torch.multinomial(probabilities, num_samples = 1)
                tokens = torch.cat([tokens, new_token.unsqueeze(0)], dim= -1) 
        detokenized_txt = self.tokenizer.detokenize(tokens[0].tolist())
        return detokenized_txt


        