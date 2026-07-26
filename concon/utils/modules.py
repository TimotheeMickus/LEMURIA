import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import math

class MultiHeadsClassifier:
    def __init__(self, image_encoder, optimizer, heads, n_heads, get_head_targets, device):
        self.image_encoder = image_encoder
        self.optimizer = optimizer
        self.heads = heads
        self.n_heads = n_heads
        self.get_head_targets = get_head_targets
        self.device = device

    def run_batch(self, batch): # Only the target signals will be used
        self.optimizer.zero_grad()

        hits, losses = self.forward(batch)

        loss = 0. # Will be the sum over all heads of the mean over the batch
        for x in losses: loss += x.mean()

        loss.backward()
        self.optimizer.step()

        return hits, loss

    def forward(self, batch): # Only the target signals will be used
        batch_img = batch.target_img(stack=True)
        activation = self.image_encoder(batch_img)
        targets = batch.target_category(stack=True, f=self.get_head_targets, device=self.device)

        losses = []
        hits = []
        for head, target in zip(self.heads, torch.unbind(targets, dim=1)):
            pred = head(activation)

            losses.append(F.nll_loss(pred, target, reduction='none'))
            hits.append((pred.argmax(dim=1) == target).float())

        return hits, losses # Lists with one element per head


# Signal -> vector
class SignalEncoder(nn.Module):
    """
    Encodes a signal of discrete symbols in a single vector.
    """
    def __init__(self, base_alphabet_size, embedding_dim, output_dim, symbol_embeddings):
        super().__init__()

        self.symbol_embeddings = symbol_embeddings
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=output_dim, num_layers=1, batch_first=True)
        
        self.alphabet_size = self.symbol_embeddings.num_embeddings

    def forward(self, signal, length):
        """
        Forward propagation.
        Input:
            `signal`, of shape [args.batch_size, <=SIGNAL_LEN], signal produced by sender
            `length`, of shape [args.batch_size, 1], length of signal produced by sender
        Output:
            encoded signal, of shape [args.batch_size, output_dim]
        """
        embeddings = self.symbol_embeddings(signal) # Shape: (batch size, signal length, embedding_dim)
        embeddings = torch.nn.utils.rnn.pack_padded_sequence(embeddings, length.squeeze(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(embeddings) # Shape: (num_layers, batch size, output_dim)
        
        return hidden[-1] # Shape: (batch size, output_dim)

    @classmethod
    def from_args(cls, args, symbol_embeddings=None):
        if(symbol_embeddings is None): symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=False)
        return cls(args.base_alphabet_size, args.hidden_size, args.hidden_size, symbol_embeddings=symbol_embeddings)

class SignalDecoder(nn.Module):
    '''
    This is a 1-layer LSTM that generates tokens autoregressively.
    The predicate embedding is projected into initial LSTM cell and hidden state.
    At each step:
      embeds the last symbol,
      runs LSTM step,
      projects to token logits (`action_space_proj`),
      samples a token (training) or argmax (eval),
      stops after producing EOS and pad rest.
    '''
    def __init__(self, base_alphabet_size, embedding_dim, output_dim, max_signal_len, symbol_embeddings):
        super().__init__()

        self.symbol_embeddings = symbol_embeddings
        self.lstm = nn.LSTM(embedding_dim, output_dim, 1)
        # project encoded signal onto cell
        self.cell_proj = nn.Linear(embedding_dim, embedding_dim)
        # project encoded signal onto hidden
        self.hidden_proj = nn.Linear(embedding_dim, embedding_dim)
        # project lstm output onto action space
        self.action_space_proj = nn.Linear(embedding_dim, base_alphabet_size + 1)

        self.max_signal_len = max_signal_len
        
        self.alphabet_size = self.symbol_embeddings.num_embeddings
        assert self.alphabet_size == (base_alphabet_size + 3) # +3: EOS symbol, padding symbol, BOS symbol
        self.eos_index = 0
        self.padding_idx = base_alphabet_size + 1
        self.bos_index = base_alphabet_size + 2 # not actually used in the signals produced

    # Returns a dictionary.
    # encoded: tensor of shape (batch size, encoding size)
    def forward(self, encoded):
        # Initialisation
        last_symbol = torch.full(size=(encoded.size(0),), fill_value=self.bos_index, device=encoded.device, dtype=torch.long)
        cell = self.cell_proj(encoded).unsqueeze(0)
        hidden = self.hidden_proj(encoded).unsqueeze(0)
        state = (hidden, cell)

        # outputs
        signal = []
        log_probs = []
        entropy = []

        # Used in the stopping mechanism (False·s become True·s when EOS is produced)
        has_stopped = torch.zeros(encoded.size(0), device=encoded.device, dtype=torch.bool)

        # Produces the signals.
        for step in range(self.max_signal_len):
            # Forces a final EOS for signals reaching the maximum length.
            if(step == (self.max_signal_len - 1)):
                forced_symbol = torch.full_like(signal[-1], self.padding_idx).masked_fill(~has_stopped, self.eos_index)
                signal.append(forced_symbol)
                log_probs.append(torch.zeros_like(has_stopped, dtype=torch.float))
                entropy.append(torch.zeros_like(has_stopped, dtype=torch.float))
                
                break

            output, state = self.lstm(self.symbol_embeddings(last_symbol).unsqueeze(0), state)
            output = self.action_space_proj(output).squeeze(0)

            # Selects actions.
            #probs = F.softmax(output, dim=-1) # Shape: (batch size, (alphabet size + 1))
            #dist = Categorical(probs)
            dist = Categorical(logits=output)
            action = dist.sample() if(self.training) else output.argmax(dim=-1) # Shape: (batch size)

            # Ignores prediction for completed signals.
            active = (~has_stopped).float()
            log_p = dist.log_prob(action) * active
            ent = dist.entropy() * active
            log_probs.append(log_p)
            entropy.append(ent)

            action = action.masked_fill(has_stopped, self.padding_idx)
            signal.append(action)

            # Stops if all signals are complete.
            has_stopped = has_stopped | (action == self.eos_index)
            if(has_stopped.all()):
                break

            last_symbol = action

        # Converts output to tensor.
        signal = torch.stack(signal, dim=1) # Shape: (batch size, max signal length)
        signal_len = (signal != self.padding_idx).sum(dim=1)[:, None] # Shape: (batch size, 1)
        log_probs = torch.stack(log_probs, dim=1) # Shape: (batch size, max signal length)

        # Averages entropy (over timesteps).
        entropy = torch.stack(entropy, dim=1) # Shape: (batch size, max signal length)
        entropy = entropy.sum(dim=1, keepdim=True) # Shape: (batch size, 1)
        entropy = entropy / signal_len.float() # The average symbol distribution entropy over the signal. Shape: (batch size, 1)

        outputs = {
            "entropy": entropy,
            "log_probs": log_probs,
            "signal": signal,
            "signal_len": signal_len}

        return outputs

    @classmethod
    def from_args(cls, args, symbol_embeddings=None):
        if(symbol_embeddings is None): symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=True)
        return cls(
            base_alphabet_size=args.base_alphabet_size,
            embedding_dim=args.hidden_size,
            output_dim=args.hidden_size,
            max_signal_len=args.max_len,
            symbol_embeddings=symbol_embeddings,)

# Adds noise to vectors.
class NoiseAdder(nn.Module):
    def __init__(self):
        super().__init__()

    # input: tensor of any shape
    # output: tensor of the same shape
    def forward(self, input):
        noise = torch.randn_like(input, device=input.device)

        return (input + noise)

    @classmethod
    def from_args(cls, args):
        return cls()
    
class CandidateNodeAverager(nn.Module):
    """
    Rudimentary candidate encoder that averages embedded node ids.
    Turns graph tensors into a fixed-size vector per candidate by embedding every node ID and averaging.
    """
    def __init__(self, node_vocab_size, hidden_size, padding_idx):
        super().__init__()

        self.node_emb = nn.Embedding(node_vocab_size, hidden_size, padding_idx=padding_idx)
        self.padding_idx = padding_idx

    def forward(self, node_idx, **_):
        '''
        Input: `node_idx` a tensor of (batch size, no. of candidates, max nodes) node IDs
        for each candidate graph.
        Output: One (B, C, H) vector per candidate where H = hidden size.
            For a candidate graph with node IDs (i1,...,in), 
            the embedding matrix E maps each node ID to a vector in H.
            For each node vector e = E[i] drop any padding nodes (mask=0) and sum the remaining vectors.
            The output is the average, i.e. the sum over the number of non-padding nodes. 
            (This is done element-wise.)
        '''
        emb = self.node_emb(node_idx) # (batch size, num candidates, max_nodes, hidden_size)
        mask = (node_idx == self.padding_idx) # (batch size, num candidates, max_nodes)
        emb = emb.masked_fill(mask.unsqueeze(-1), 0.0) # (batch size, num candidates, max_nodes, hidden_size)
        summed = emb.sum(dim=2) # (batch size, num candidates, hidden_size)
        counts = (~mask).sum(dim=2) # (batch size, num candidates)
        return summed / counts.unsqueeze(-1) # (batch, num_candidates, hidden)
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model: Int
        # num_heads: Int

        super().__init__()

        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads # output dim of each head
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.e_linear = nn.Linear(d_model, self.num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, e, attn_mask):
        # q: tensor of shape [batch size, num heads, num nodes, d_head]
        # k: tensor of shape [batch size, num heads, num nodes, d_head]
        # v: tensor of shape [batch size, num heads, num nodes, d_head]
        # e: None | tensor of shape [batch size, num heads, num nodes, num nodes]
        # attn_mask: None | boolean tensor of shape [batch size, num heads, num nodes, num nodes], indicates for each head and each pair of nodes whether the first can attend to the second (True) or not (False)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head) # [batch size, num heads, num nodes, num nodes]
        if(e is not None): scores = scores + e # [batch size, num heads, num nodes, num nodes]
        
        if(attn_mask is not None): scores = scores.masked_fill((attn_mask == False), float('-inf'))
        
        probs = torch.softmax(scores, dim=-1) # [batch size, num heads, num nodes, num nodes]
        if(attn_mask is not None): probs = probs.masked_fill((attn_mask == False), 0.0)
        #print(probs) # DEBUG

        return torch.matmul(probs, v) # [batch size, num heads, num nodes, d_head]

    def forward(self, node_emb, edge_emb, attn_mask):
        # node_emb: tensor of shape [batch size, num nodes, d_model]
        # edge_emb: None | tensor of shape [batch size, num nodes, num nodes, d_model]
        # attn_mask: None | boolean tensor of shape [batch size, num heads, num nodes, num nodes], indicates for each head and each pair of nodes whether the first can attend to the second (True) or not (False)

        batch_size = node_emb.size(0)
        num_nodes = node_emb.size(1)

        # Computes queries, keys and values.
        q = self.q_linear(node_emb).view(batch_size, num_nodes, self.num_heads, self.d_head).transpose(1, 2) # [batch size, num heads, num nodes, d_head]
        k = self.k_linear(node_emb).view(batch_size, num_nodes, self.num_heads, self.d_head).transpose(1, 2) # [batch size, num heads, num nodes, d_head]
        v = self.v_linear(node_emb).view(batch_size, num_nodes, self.num_heads, self.d_head).transpose(1, 2) # [batch size, num heads, num nodes, d_head]

        # Computes edge scores. (Each head cares differently about different types of edge, but without taking the nodes involved into account.)
        if(edge_emb is not None): e = self.e_linear(edge_emb).permute(0, 3, 1, 2) # [batch size, num heads, num nodes, num nodes]
        else: e = None

        # Applies attention.
        vectors = self.attention(q, k, v, e, attn_mask) # [batch size, num heads, num nodes, d_head]
        #print(vectors) # DEBUG
        # Concatenates the output of all heads.
        concat = vectors.transpose(1, 2).reshape(batch_size, num_nodes, (self.num_heads * self.d_head)) # [batch size, num nodes, d_model]
        # Applies the final linear projection.
        return self.out_linear(concat) # [batch size, num nodes, d_model]


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout, use_norm):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model) if(use_norm) else None # Because `d_model` is a scalar, each vector is normalised individually.
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.norm2 = nn.LayerNorm(d_model) if(use_norm) else None # Because `d_model` is a scalar, each vector is normalised individually.

        self.dropout = nn.Dropout(dropout)

    def num_heads(self):
        return self.self_attn.num_heads

    def forward(self, node_emb, edge_emb=None, node_mask=None, attn_mask=None):
        # node_emb: tensor of shape [batch size, num nodes, d_model]
        # edge_emb: None | tensor of shape [batch size, num nodes, num nodes, d_model]
        # node_mask: None | boolean tensor of shape [batch size, num nodes], indicates for each node whether it is an actual node (True) or not (False)
        # attn_mask: None | boolean tensor of shape [batch size, num heads, num nodes, num nodes], indicates for each head and each pair of nodes whether the first can attend to the second (True) or not (False)

        # Masks padding nodes. This is useful (or not?) to avoid problems in the attention layer.
        #if(node_mask is not None): node_emb = node_emb.masked_fill((node_mask == False).unsqueeze(-1), 0.0) # [batch size, num nodes, d_model]

        # Self-attention
        attn_output = self.self_attn(node_emb=node_emb, edge_emb=edge_emb, attn_mask=attn_mask) # [batch size, num nodes, d_model]

        # if(node_mask is not None): attn_output = attn_output.masked_fill((node_mask == False).unsqueeze(-1), 0.0) # [batch size, num nodes, d_model] # DEBUG

        # Dropout + residual connection + normalisation
        x = node_emb + self.dropout(attn_output) # [batch size, num nodes, d_model]
        if(self.norm1 is not None): x = self.norm1(x) # [batch size, num nodes, d_model]

        # Feed-forward
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(x)))) # [batch size, num nodes, d_model]

        # Dropout + residual connection + normalisation
        y = x + self.dropout(ff_output) # [batch size, num nodes, d_model]
        if(self.norm2 is not None): y = self.norm2(y) # [batch size, num nodes, d_model]

        # Masks padding nodes. This is useful (or not?) to avoid problems in possible subsequent attention layers.
        #if(node_mask is not None): y = y.masked_fill((node_mask == False).unsqueeze(-1), 0.0) # [batch size, num nodes, d_model]

        return y # [batch size, num nodes, d_model]


class GraphTransformerEncoder(nn.Module):
    def __init__(self, node_vocab_size, edge_vocab_size, num_layers, d_model, num_heads, d_hidden, dropout, use_norm=True):
        super().__init__()
        self.node_embedding = nn.Embedding(node_vocab_size, d_model)
        self.edge_embedding = nn.Embedding(edge_vocab_size, d_model)
        self.layers = nn.ModuleList([GraphTransformerEncoderLayer(d_model, num_heads, d_hidden, dropout, use_norm) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def num_heads(self):
        return self.layers[0].num_heads()

    def forward(self, node_idx, edge_idx, graph_size=None, attn_mask=None):
        '''
        Input:
            node_idx: (B, N) node IDs for each node
            edge_idx: (B, N, N) edge label IDs
            graph_size: None | (B,) number of valid nodes per graph
            attn_mask: None | (B, H, N, N) attention mask for the transformer heads
        Output:
            (B, N, D) candidate embeddings, where D = d_model.

        The encoder embeds nodes and edges, applies L graph-transformer layers, yields one vector per node.
        '''
        # node_idx: tensor of shape [batch size, num nodes]
        # edge_idx: tensor of shape [batch size, num nodes, num nodes]
        # graph_size: None | int tensor of shape [batch size]
        # attn_mask: None | boolean tensor of shape [batch size, num heads, num nodes, num nodes], indicates for each head and each pair of nodes whether the first can attend to the second (True) or not (False)
        batch_size = node_idx.size(0)
        num_nodes = node_idx.size(1)

        if(graph_size is not None):
          # Creates a boolean mask indicating where are the actual nodes. (True ~ actual node)
          node_mask = torch.arange(num_nodes, device=node_idx.device)[None, :] < graph_size[:, None] # [batch size, num nodes]

          # Creates a boolean mask indicating that padding nodes cannot attend to or be attended by any node. (True ~ attention allowed)
          row_idx = torch.arange(num_nodes, device=node_idx.device).view(1, -1, 1) # [1, num nodes, 1]
          col_idx = torch.arange(num_nodes, device=node_idx.device).view(1, 1, -1) # [1, 1, num nodes]
          graph_size = graph_size.view(-1, 1, 1) # [batch size, 1, 1]
          tmp_mask = (row_idx < graph_size) & (col_idx < graph_size) # [batch size, num nodes, num nodes]
          tmp_mask = tmp_mask.unsqueeze(1).expand(batch_size, self.num_heads(), num_nodes, num_nodes) # [batch size, num heads, num nodes, num nodes]

          if(attn_mask is None): attn_mask = tmp_mask # [batch size, num heads, num nodes, num nodes]
          else: attn_mask = attn_mask & tmp_mask # [batch size, num heads, num nodes, num nodes]
        else: node_mask = None

        node_emb = self.node_embedding(node_idx) # [batch size, num nodes, d_model]
        node_emb = self.dropout(node_emb) # [batch size, num nodes, d_model]

        edge_emb = self.edge_embedding(edge_idx) # [batch size, num nodes, num nodes, d_model]

        for layer in self.layers:
            node_emb = layer(node_emb=node_emb, edge_emb=self.dropout(edge_emb), node_mask=node_mask, attn_mask=attn_mask) # [batch size, num nodes, d_model]
            
        return node_emb # [batch size, num nodes, d_model]
    

class CandidateGraphEncoder(nn.Module):
    def __init__(self, node_vocab_size, edge_vocab_size, num_layers, d_model, num_heads, d_hidden, dropout, use_norm=True):
        super().__init__()
        self.graph_encoder = GraphTransformerEncoder(node_vocab_size, edge_vocab_size, num_layers, d_model, num_heads, d_hidden, dropout, use_norm)

    def forward(self, node_idx, edge_idx, graph_size=None):
        '''
        Input:
            node_idx: (B, C, N) node IDs for each candidate graph
            edge_idx: (B, C, N, N) edge label IDs
            graph_size: None | (B, C) number of valid nodes per graph
        Output:
            (B, C, D) 
        '''
        # Flattens the batch and candidate dimensions.
        batch_size = node_idx.size(0)
        num_candidates = node_idx.size(1)
        num_nodes = node_idx.size(2)

        flat_node_idx = node_idx.view(batch_size * num_candidates, num_nodes) # [batch size * num_candidates, num_nodes]
        flat_edge_idx = edge_idx.view(batch_size * num_candidates, num_nodes, num_nodes) # [batch size * num_candidates, num_nodes, num_nodes]
        flat_graph_size = graph_size.view(-1) if graph_size is not None else None # [batch size * num_candidates] | None

        node_emb = self.graph_encoder(flat_node_idx, flat_edge_idx, flat_graph_size, attn_mask=None) # [batch size * num_candidates, num nodes, d_model]

        if graph_size is None:
            pooled = node_emb.mean(dim=1) # [batch size * num_candidates, d_model]
        else:
            # Creates a boolean mask indicating where are the actual nodes. (True ~ actual node)
            node_mask = torch.arange(num_nodes, device=flat_node_idx.device)[None, :] < flat_graph_size[:, None] # [batch size * num_candidates, num nodes]
            node_emb = node_emb.masked_fill((node_mask == False).unsqueeze(-1), 0.0) # [batch size * num_candidates, num nodes, d_model]

            pooled = node_emb.sum(dim=1) # [batch size * num_candidates, d_model]
            pooled = pooled / flat_graph_size.unsqueeze(-1) # [batch size * num_candidates, d_model]
        return pooled.view(batch_size, num_candidates, -1) # [batch, num_candidates, d_model]


# output: torch.nn.Module
# hidden_size: int
# track_running_stats: bool, pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added)
def _dcgan_tuto_cnn(hidden_size, track_running_stats=True):
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size) x 32 x 32
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*2) x 16 x 16
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*4) x 8 x 8
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*4) x 8 x 8
            nn.Conv2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*8) x 4 x 4
            nn.Conv2d(hidden_size, hidden_size, 4, 1, 0, bias=False),
            nn.Tanh(),
            nn.Flatten(),
        )

# output: torch.nn.Module
# hidden_size: int
# track_running_stats: bool, pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added)
def _dcgan_tuto_decnn(hidden_size, track_running_stats=True):
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            nn.ConvTranspose2d(hidden_size, hidden_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            nn.ConvTranspose2d(hidden_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 128 x 128
        )

# output: torch.nn.Module
# hidden_size: int
# track_running_stats: bool, pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added)
def _dcgan_decnn(hidden_size, track_running_stats=True):
    """A more viable CNN/DCNN architecture"""
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is Z, going into a convolution
            #1
            nn.Upsample(scale_factor=4, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size, hidden_size * 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 16, track_running_stats=track_running_stats),
            nn.ReLU(True),
             # state size. (hidden_size*16) x 4 x 4
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size * 16, hidden_size * 8, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 8, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*8) x 8 x 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size * 8, hidden_size * 4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 4, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*4) x 16 x 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size * 4, hidden_size *2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size * 2, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size*2) x 32 x 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size * 2, hidden_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size, track_running_stats=track_running_stats),
            nn.ReLU(True),
            # state size. (hidden_size) x 64 x 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_size, 3, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

# output: torch.nn.Module
# hidden_size: int
# track_running_stats: bool, pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added)
def _dcgan_cnn(hidden_size, track_running_stats=True):
    """A more viable CNN/DCNN architecture"""
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(3, hidden_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (hidden_size) x 64 x 64
            nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 2, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (hidden_size) x 32 x 32
            nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 4, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (hidden_size*2) x 16 x 16
            nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 8, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (hidden_size*4) x 8 x 8
            nn.Conv2d(hidden_size * 8, hidden_size * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_size * 16, track_running_stats=track_running_stats),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (hidden_size*8) x 4 x 4
            nn.Conv2d(hidden_size * 16, hidden_size, 4, 1, 0, bias=False),
            nn.Flatten(),
        )


# def build_cnn(layer_classes=(), input_channels=(), output_channels=(),
#     strides=(), kernel_size=None, paddings=None, flatten_last=True,
#     sigmoid_after=False,):
#     """
#     Factory for convolutionnal encoders.
#     Input:
#         `layer_classes`: a list of classes to stack, taken from `{"conv", "convTranspose", "maxpool", "avgpool"}`
#         `input_channels`: a list of expected input channels per layer
#         `output_channels`: a list of expected output channels per layer
#         `strides`: a list of strides per layer each layer
#         `kernel_size`: a valid kernel size used throughout the convolutionnal network encoder, or a list of kernel sizes per layer
#         `padding`: an optional list of (output) padding per layer
#         `flatten_last`: flatten output instead of performing batch normalization after the last layer.
#     Output:
#         `cnn`: a convolutionnal network
#     Raises:
#         `AssertionError` if the provided lists `layer_classes`, `input_channels`, `output_channels`, and `strides` have different lengths
#         `ValueError` if a given layer class is not "conv", "maxpool", or "avgpool"
#     """
#
#     lens = map(len, (layer_classes, input_channels, output_channels, strides))
#     assert len(set(lens)) == 1, "provided parameters have different lengths!"
#
#     if paddings is None:
#         paddings = ([0] * len(layer_classes))
#     else:
#         assert len(layer_classes) == len(paddings), "provided parameters have different lengths!"
#
#     if (type(kernel_size) is int) or (len(kernel_size) == 2):
#         kernel_size = ([kernel_size] * len(layer_classes))
#     else:
#         assert len(layer_classes) == len(kernel_size), "provided parameters have different lengths!"
#
#     if flatten_last:
#         norms = ([nn.BatchNorm2d] * (len(layer_classes) - 1)) + [lambda _ : nn.Flatten()]
#     else:
#         norms = ([nn.BatchNorm2d] * len(layer_classes))
#
#     layers = []
#
#     for s,i,o,n,l,p,k in zip(
#         strides,
#         input_channels,
#         output_channels,
#         norms,
#         layer_classes,
#         paddings,
#         kernel_size,):
#         if l == "conv":
#             core_layer = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=i,
#                     out_channels=o,
#                     kernel_size=k,
#                     stride=s,
#                     padding=p,),
#                 nn.ReLU())
#         elif l == "convTranspose":
#             core_layer = nn.Sequential(
#                 nn.ConvTranspose2d(
#                     in_channels=i,
#                     out_channels=o,
#                     kernel_size=k,
#                     stride=s,
#                     output_padding=p,),
#                 nn.ReLU())
#         elif l == "maxpool":
#             core_layer = nn.MaxPool2d(
#                 kernel_size=k,
#                 stride=s,
#                 padding=p,)
#         elif l == "avgpool":
#             core_layer = nn.AvgPool2d(
#                 kernel_size=k,
#                 stride=s,
#                 padding=p,)
#         else:
#             raise ValueError("layer of type %s is not supported.")
#         layers.append(
#             nn.Sequential(
#                 core_layer,
#                 n(o),
#         ))
#     if sigmoid_after:
#         layers.append(nn.Sigmoid())
#     cnn = nn.Sequential(*layers)
#     return cnn

# output: torch.nn.Module
def build_cnn_encoder_from_args(args):
    """
    Factory for convolutionnal networks
    """
    features = args.cnn_channel_size or args.hidden_size
    track_running_stats = (not args.local_batchnorm) # Pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added).
    
    if(args.use_legacy_cnn or args.use_legacy_convolutions):
        if(not args.quiet): print("Using legacy convolution architecture")
        short_cut = _dcgan_tuto_cnn(features, track_running_stats=track_running_stats)
    else:
        if(not args.quiet): print("Using modern convolution architecture")
        short_cut = _dcgan_cnn(features, track_running_stats=track_running_stats)
    
    return short_cut
    #
    # # for legacy or now
    # layer_classes = (["conv"] * args.conv_layers)
    # input_channels = ([args.img_channel] + [args.filters] * (args.conv_layers - 1))
    # output_channels = ([args.filters] * (args.conv_layers - 1) + [args.hidden_size])
    # return build_cnn(
    #     layer_classes=layer_classes,
    #     input_channels=input_channels,
    #     output_channels=output_channels,
    #     strides=args.strides,
    #     kernel_size=args.kernel_size,
    #     paddings=None,)

# output: torch.nn.Module
def build_cnn_decoder_from_args(args):
    """
    Factory for deconvolutionnal networks
    """
    features = args.decnn_channel_size or args.hidden_size
    track_running_stats = (not args.local_batchnorm) # Pertains to torch.nn.BatchNorm2d (if True, hidden statistics tracking parameters are added).

    if(args.use_legacy_decnn or args.use_legacy_convolutions):
        if(not args.quiet): print("Using legacy deconvolution architecture")
        short_cut = _dcgan_tuto_decnn(features, track_running_stats=track_running_stats)
    else:
        if(not args.quiet): print("Using modern deconvolution architecture")
        short_cut = _dcgan_decnn(features, track_running_stats=track_running_stats)
    
    return short_cut

    # #for legacy for now
    # layer_classes = (["convTranspose"] * args.conv_layers)
    # strides = args.strides[::-1]
    # inputs = [args.hidden_size] + ([args.filters] * (args.conv_layers - 1))
    # outputs = ([args.filters] * (args.conv_layers - 1)) + [args.img_channel]
    # paddings = [0, 0, 1, 0, 0, 0, 0, 1] # guessworking it out
    # return build_cnn(
    #     layer_classes=layer_classes,
    #     input_channels=inputs,
    #     output_channels=outputs,
    #     strides=strides,
    #     paddings=paddings,
    #     kernel_size=args.kernel_size,
    #     flatten_last=False,
    #     sigmoid_after=True,)

# output: torch.nn.Embedding
def build_embeddings(base_alphabet_size, dim, use_bos=False):
    vocab_size = (base_alphabet_size + 3) if use_bos else (base_alphabet_size + 2) # +3: EOS symbol, padding symbol, BOS symbol; +2: EOS symbol, padding symbol
    return nn.Embedding(vocab_size, dim, padding_idx=base_alphabet_size + 1)
