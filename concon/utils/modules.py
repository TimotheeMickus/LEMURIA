import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class MultiHeadsClassifier:
    def __init__(self, image_encoder, optimizer, heads, n_heads, get_head_targets, device):
        self.image_encoder = image_encoder
        self.optimizer = optimizer
        self.heads = heads
        self.n_heads = n_heads
        self.get_head_targets = get_head_targets
        self.device = device

    def train(self, batch): # Only the target images will be used
        self.optimizer.zero_grad()

        hits, losses = self.forward(batch)

        loss = 0. # Will be the sum over all heads of the mean over the batch
        for x in losses: loss += x.mean()

        loss.backward()
        self.optimizer.step()

        return hits, loss

    def forward(self, batch): # Only the target images will be used
        batch_img = batch.target_img(stack=True)
        activation = self.image_encoder(batch_img)
        targets = batch.category(stack=True, f=self.get_head_targets).to(self.device)

        losses = []
        hits = []
        for head, target in zip(self.heads, torch.unbind(targets, dim=1)):
            pred = head(activation)

            losses.append(F.nll_loss(pred, target, reduction='none'))
            hits.append((pred.argmax(dim=1) == target).float())

        return hits, losses # Lists with one element per head

# Message -> vector
class MessageEncoder(nn.Module):
    """
    Encodes a message of discrete symbols in a single vector.
    """
    def __init__(self,
        base_alphabet_size,
        embedding_dim,
        output_dim,
        symbol_embeddings):
        super(MessageEncoder, self).__init__()

        self.symbol_embeddings = symbol_embeddings

        self.lstm = nn.LSTM(embedding_dim, output_dim, 1, batch_first=True)

    def forward(self, message, length):
        """
        Forward propagation.
        Input:
            `message`, of shape [args.batch_size x <=MSG_LEN], message produced by sender
            `length`, of shape [args.batch_size x 1], length of message produced by sender
        Output:
            encoded message, of shape [args.batch_size x output_dim]
        """
        # encode
        embeddings = self.symbol_embeddings(message)
        embeddings = self.lstm(embeddings)[0]
        # select last step corresponding to message
        index = torch.arange(message.size(-1)).expand_as(message).to(message.device)
        output = embeddings.masked_select((index == (length-1)).unsqueeze(-1))

        return output.view(embeddings.size(0), embeddings.size(-1))

    @classmethod
    def from_args(cls, args, symbol_embeddings=None):
        if(symbol_embeddings is None): symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=False)
        return cls(args.base_alphabet_size, args.hidden_size, args.hidden_size, symbol_embeddings=symbol_embeddings)

# Vector -> message
class MessageDecoder(nn.Module):
    def __init__(self,
        base_alphabet_size,
        embedding_dim,
        output_dim,
        max_msg_len,
        symbol_embeddings,
        ):
        super(MessageDecoder, self).__init__()

        self.symbol_embeddings = symbol_embeddings

        self.lstm = nn.LSTM(embedding_dim, output_dim, 1)
        # project encoded img onto cell
        self.cell_proj = nn.Linear(embedding_dim, embedding_dim)
        # project encoded img onto hidden
        self.hidden_proj = nn.Linear(embedding_dim, embedding_dim)
        # project lstm output onto action space
        self.action_space_proj = nn.Linear(embedding_dim, base_alphabet_size + 1)

        self.max_msg_len = max_msg_len
        self.bos_index = base_alphabet_size + 2
        self.eos_index = 0
        self.padding_idx = base_alphabet_size + 1

    def forward(self, encoded):
        # Initialisation
        last_symbol = torch.ones(encoded.size(0)).long().to(encoded.device) * self.bos_index
        cell = self.cell_proj(encoded).unsqueeze(0)
        hidden = self.hidden_proj(encoded).unsqueeze(0)
        state = (cell, hidden)

        # outputs
        message = []
        log_probs = []
        entropy = []

        # Used in the stopping mechanism (when EOS has been produced)
        has_stopped = torch.zeros(encoded.size(0)).bool().to(encoded.device)
        has_stopped.requires_grad = False

        # Produces the messages
        # TODO Je serais d'avis à ne pas utiliser de EOS. Si l'action EOS est choisie, le message serait terminé sans qu'aucun symbol ne soit ajouté (ou plus techniquement, on ajoute un padding symbol). En fait, ça revient plus ou moins à fusionner le EOS et le padding symbol. Cela permettrait d'éviter d'avoir un symbol spécial apparaissant souvent mais pas toujours dans les "vrais" messages, ce qui peut compliquer l'analyse.
        for i in range(self.max_msg_len):
            output, state = self.lstm(self.symbol_embeddings(last_symbol).unsqueeze(0), state)
            output = self.action_space_proj(output).squeeze(0)

            # Selects actions
            probs = F.softmax(output, dim=-1)
            dist = Categorical(probs)
            action = dist.sample() if self.training else probs.argmax(dim=-1)

            # Ignores prediction for completed messages
            ent = dist.entropy() * (~has_stopped).float()
            log_p = dist.log_prob(action) * (~has_stopped).float()
            log_probs.append(log_p)
            entropy.append(ent)

            action = action.masked_fill(has_stopped, self.padding_idx)
            message.append(action)

            # Stops if all messages are complete
            has_stopped = has_stopped | (action == self.eos_index)
            if has_stopped.all():
                break

            last_symbol = action

        # Converts output to tensor
        message = torch.stack(message, dim=1)
        message_len = (message != self.padding_idx).sum(dim=1)[:,None]
        log_probs = torch.stack(log_probs, dim=1)

        # Average entropy over timesteps, hence ignore padding
        entropy = torch.stack(entropy, dim=1)
        entropy = entropy.sum(dim=1, keepdim=True)
        entropy = entropy / message_len.float() # The average symbol distribution entropy over the message

        outputs = {
            "entropy": entropy,
            "log_probs": log_probs,
            "message": message,
            "message_len": message_len}

        return outputs

    @classmethod
    def from_args(cls, args, symbol_embeddings=None):
        if(symbol_embeddings is None): symbol_embeddings = build_embeddings(args.base_alphabet_size, args.hidden_size, use_bos=True)
        return cls(
            base_alphabet_size=args.base_alphabet_size,
            embedding_dim=args.hidden_size,
            output_dim=args.hidden_size,
            max_msg_len=args.max_len,
            symbol_embeddings=symbol_embeddings,)

# vector -> vector + random noise
class Randomizer(nn.Module):
    def __init__(self, input_dim, random_dim):
        super(Randomizer, self).__init__()
        self.merging_projection = nn.Linear(input_dim + random_dim, input_dim)
        self.random_dim = random_dim
        self.input_dim = input_dim

    def forward(self, input_vector):
        """
        Input:
            `input_vector` of dimension [BATCH x self.input_dim]
        """
        noise = torch.randn(input_vector.size(0), self.random_dim, device=input_vector.device)
        input_with_noise = torch.cat([input_vector, noise], dim=1)
        merged_input = self.merging_projection(input_with_noise)
        return merged_input

    @classmethod
    def from_args(cls, args):
        return cls(input_dim=args.hidden_size, random_dim=args.hidden_size)

def _dcgan_tuto_cnn(img_size, hidden_size):
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is (3) x 64 x 64
            nn.Conv2d(3, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(img_size * 8, hidden_size, 4, 1, 0, bias=False),
            nn.Tanh()
        )


def _dcgan_tuto_decnn(img_size, hidden_size):
    # params of convs:
    # input chans, output chans, kernel, stride, padding
    # ignore coms which are incorrect wrt. our IMG size
    return nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(hidden_size, img_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(img_size * 8, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( img_size * 4, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( img_size * 2, img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(img_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (3) x 128 x 128
        )

def build_cnn(layer_classes=(), input_channels=(), output_channels=(),
    strides=(), kernel_size=None, paddings=None, flatten_last=True,
    sigmoid_after=False,):
    """
    Factory for convolutionnal encoders.
    Input:
        `layer_classes`: a list of classes to stack, taken from `{"conv", "convTranspose", "maxpool", "avgpool"}`
        `input_channels`: a list of expected input channels per layer
        `output_channels`: a list of expected output channels per layer
        `strides`: a list of strides per layer each layer
        `kernel_size`: a valid kernel size used throughout the convolutionnal network encoder, or a list of kernel sizes per layer
        `padding`: an optional list of (output) padding per layer
        `flatten_last`: flatten output instead of performing batch normalization after the last layer.
    Output:
        `cnn`: a convolutionnal network
    Raises:
        `AssertionError` if the provided lists `layer_classes`, `input_channels`, `output_channels`, and `strides` have different lengths
        `ValueError` if a given layer class is not "conv", "maxpool", or "avgpool"
    """

    lens = map(len, (layer_classes, input_channels, output_channels, strides))
    assert len(set(lens)) == 1, "provided parameters have different lengths!"

    if paddings is None:
        paddings = ([0] * len(layer_classes))
    else:
        assert len(layer_classes) == len(paddings), "provided parameters have different lengths!"

    if (type(kernel_size) is int) or (len(kernel_size) == 2):
        kernel_size = ([kernel_size] * len(layer_classes))
    else:
        assert len(layer_classes) == len(kernel_size), "provided parameters have different lengths!"

    if flatten_last:
        norms = ([nn.BatchNorm2d] * (len(layer_classes) - 1)) + [lambda _ : nn.Flatten()]
    else:
        norms = ([nn.BatchNorm2d] * len(layer_classes))

    layers = []

    for s,i,o,n,l,p,k in zip(
        strides,
        input_channels,
        output_channels,
        norms,
        layer_classes,
        paddings,
        kernel_size,):
        if l == "conv":
            core_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=k,
                    stride=s,
                    padding=p,),
                nn.ReLU())
        elif l == "convTranspose":
            core_layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=k,
                    stride=s,
                    output_padding=p,),
                nn.ReLU())
        elif l == "maxpool":
            core_layer = nn.MaxPool2d(
                kernel_size=k,
                stride=s,
                padding=p,)
        elif l == "avgpool":
            core_layer = nn.AvgPool2d(
                kernel_size=k,
                stride=s,
                padding=p,)
        else:
            raise ValueError("layer of type %s is not supported.")
        layers.append(
            nn.Sequential(
                core_layer,
                n(o),
        ))
    if sigmoid_after:
        layers.append(nn.Sigmoid())
    cnn = nn.Sequential(*layers)
    return cnn

def build_cnn_encoder_from_args(args):
    """
    Factory for convolutionnal networks
    """
    short_cut = _dcgan_tuto_cnn(128, args.hidden_size)
    return short_cut

    # for legacy or now
    layer_classes = (["conv"] * args.conv_layers)
    input_channels = ([args.img_channel] + [args.filters] * (args.conv_layers - 1))
    output_channels = ([args.filters] * (args.conv_layers - 1) + [args.hidden_size])
    return build_cnn(
        layer_classes=layer_classes,
        input_channels=input_channels,
        output_channels=output_channels,
        strides=args.strides,
        kernel_size=args.kernel_size,
        paddings=None,)

def build_cnn_decoder_from_args(args):
    """
    Factory for deconvolutionnal networks
    """
    short_cut = _dcgan_tuto_decnn(128, args.hidden_size)
    return short_cut

    layer_classes = (["convTranspose"] * args.conv_layers)
    strides = args.strides[::-1]
    inputs = [args.hidden_size] + ([args.filters] * (args.conv_layers - 1))
    outputs = ([args.filters] * (args.conv_layers - 1)) + [args.img_channel]
    paddings = [0, 0, 1, 0, 0, 0, 0, 1] # guessworking it out
    return build_cnn(
        layer_classes=layer_classes,
        input_channels=inputs,
        output_channels=outputs,
        strides=strides,
        paddings=paddings,
        kernel_size=args.kernel_size,
        flatten_last=False,
        sigmoid_after=True,)

def build_embeddings(base_alphabet_size, dim, use_bos=False):
    vocab_size = (base_alphabet_size + 3) if use_bos else (base_alphabet_size + 2) # +3: EOS symbol, padding symbol, BOS symbol; +2: EOS symbol, padding symbol
    return nn.Embedding(vocab_size, dim, padding_idx=base_alphabet_size + 1)
