from datetime import datetime
import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.log import Progress
from ..utils.misc import build_optimizer, Unflatten
from ..utils.modules import MultiHeadsClassifier

# Mixin providing CNN-pretraining capability to *image* games.
#
# Only games whose agents carry an `image_encoder` (and optionally an
# `image_decoder`) can pretrain their convolutional stack, so this behaviour
# lives here rather than on the game base class. A game opts in simply by
# inheriting from `CNNPretrainable` (e.g. `class AliceBob(CNNPretrainable, Game)`);
# games without images (e.g. AlexBeth) do not inherit it, and the training
# driver checks `isinstance(game, CNNPretrainable)` before pretraining.
#
# Concrete image games must provide `agents_for_CNN_pretraining`.
#
# Expected on the host game: `self.autologger`.
class CNNPretrainable:
    # Lists the agents whose CNN should be pretrained. Must be overridden by the
    # concrete image game (e.g. AliceBob returns its sender/receiver).
    def agents_for_CNN_pretraining(self):
        raise NotImplementedError

    # TODO: pretraining should be handled by a separate, dedicated object
    def pretrain_CNNs(self, data_iterator, pretrain_CNN_mode='category-wise', freeze_pretrained_CNN=False, learning_rate=0.0001, epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, deconvolution_factory=None, convolution_factory=None):
        pretrained_models = {}
        agents = self.agents_for_CNN_pretraining()
        for i, agent in enumerate(agents):
            agent_name = f"agent {i}"
            pretrained_models[agent_name] = self.pretrain_agent_CNN(agent, data_iterator, pretrain_CNN_mode, freeze_pretrained_CNN, learning_rate, epochs, steps_per_epoch, display_mode, pretrain_CNNs_on_eval, deconvolution_factory, convolution_factory, agent_name=agent_name)

        return pretrained_models

    def pretrain_agent_CNN(self, agent, data_iterator, pretrain_CNN_mode='category-wise', freeze_pretrained_CNN=False, learning_rate=0.0001, epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, deconvolution_factory=None, convolution_factory=None, agent_name="agent"):
        """
        Pretrain (de)convolution of agent.
        """
        print(("[%s] pretraining %s…" % (datetime.now(), agent_name)), flush=True)

        if(pretrain_CNN_mode != 'auto-encoder'):
            pretrained_model = self._pretrain_classif(agent, data_iterator, pretrain_CNN_mode, learning_rate, epochs, steps_per_epoch, display_mode, pretrain_CNNs_on_eval, agent_name)
        else:
            pretrained_model = self._pretrain_ae(agent, data_iterator, pretrain_CNN_mode, deconvolution_factory, convolution_factory, learning_rate, epochs, steps_per_epoch, display_mode, pretrain_CNNs_on_eval, agent_name)

        # If necessary, deactivate training in the image encoder.
        if(freeze_pretrained_CNN):
            for p in agent.image_encoder.parameters():
                p.requires_grad = False

        return pretrained_model

    # Pretrains the CNN of an agent in category- or feature-wise mode
    def _pretrain_classif(self, agent, data_iterator, pretrain_CNN_mode='category-wise', learning_rate=0.0001, epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, agent_name="agent"):
        loss_tag = 'pretrain/loss_%s_%s' % (agent_name, pretrain_CNN_mode)
        acc_tag = 'pretrain/acc_%s_%s' % (agent_name, pretrain_CNN_mode)

        concept_sizes = [len(concept) for concept in data_iterator.concepts]
        xcoder = agent.signal_decoder if hasattr(agent, 'signal_decoder') else agent.signal_encoder
        hidden_size = xcoder.symbol_embeddings.weight.size(1)
        device = next(agent.parameters()).device

        if pretrain_CNN_mode == 'feature-wise':
            # Defines one classification head per non-unary concept
            heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(50, 3), # I guess the 3 could be replaced with csize
                    nn.LogSoftmax(dim=1)
                ) for csize in concept_sizes if csize > 1
            ]).to(device)
            get_head_targets = (lambda cat: [v for v, csize in zip(cat, concept_sizes) if csize > 1])
        else:
            heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(50, data_iterator.nb_categories),
                    nn.LogSoftmax(dim=1))
                ]
            ).to(device)
            get_head_targets = (lambda cat: [data_iterator.category_idx(cat)])

        optimizer = build_optimizer(it.chain(agent.image_encoder.parameters(), heads.parameters()), learning_rate)
        n_heads = len(heads)

        model = MultiHeadsClassifier(agent.image_encoder, optimizer, heads, n_heads, get_head_targets, device)

        total_items = 0
        for epoch_index in range(epochs):
            # Optimization
            pbar = Progress.get_progress_cls(display_mode)(steps_per_epoch, epoch_index, logged_items={'L', 'acc'})
            epoch_hits, epoch_items = 0., 0. # TODO Do they need to be floats instead of integers?
            with pbar:
                for step_i in range(steps_per_epoch):
                    batch = data_iterator.get_batch(data_type='train', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[]) # For each instance of the batch, one original and one target image, but no distractor; only the target will be used

                    hits, loss = model.run_batch(batch)

                    for x in hits: epoch_hits += x.sum().item()
                    epoch_items += batch.size
                    total_items += batch.size

                    if(self.autologger.summary_writer is not None):
                        self.autologger.summary_writer.add_scalar(loss_tag, (loss.item() / batch.size), total_items)
                        self.autologger.summary_writer.add_scalar(acc_tag, (epoch_hits / (epoch_items * n_heads)), total_items)
                    pbar.update(L=loss.item(), acc=(epoch_hits / (epoch_items * n_heads)))

            # Evaluation
            with torch.no_grad():
                agent.image_encoder.eval()
                heads.eval()
                test_loss, test_epoch_hits, test_epoch_items =  0., 0, 0
                for step_i in range(1 + (steps_per_epoch // 10)):
                    batch = data_iterator.get_batch(data_type='test', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[]) # For each instance of the batch, one original and one target image, but no distractor; only the target will be use
                    hits, losses = model.forward(batch)
                    for x in losses: test_loss += x.sum().item()
                    for x in hits: test_epoch_hits += x.sum().item()
                    test_epoch_items += batch.size
                test_loss = test_loss / (test_epoch_items * n_heads)
                test_acc = test_epoch_hits / (test_epoch_items * n_heads)
                if(self.autologger.summary_writer is not None):
                    self.autologger.summary_writer.writer.add_scalar('eval-' + loss_tag, test_loss, epoch_index)
                    self.autologger.summary_writer.writer.add_scalar('eval-' + acc_tag, test_acc, epoch_index)
                print(f"[eval-pretrain {agent_name} epoch {epoch_index}] L={test_loss}, acc={test_acc}")
                agent.image_encoder.train()
                heads.train()

        return model

    # Pretrains the CNN of an agent in auto-encoder mode
    def _pretrain_ae(self, agent, data_iterator, pretrain_CNN_mode='auto-encoder', deconvolution_factory=None, convolution_factory=None, learning_rate=0.0001, epochs=5, steps_per_epoch=1000, display_mode='', pretrain_CNNs_on_eval=False, agent_name="agent", _is_external_ae=False, device=None):
        loss_tag = 'pretrain/loss_%s_%s' % (agent_name, pretrain_CNN_mode)
        if device is None:
            device = next(agent.parameters()).device

        found = False
        if agent is not None and hasattr(agent, 'image_encoder'):
            encoder = agent.image_encoder
            found = True
        else:
            encoder = convolution_factory()
        if agent is not None and hasattr(agent, 'image_decoder'):
            decoder = agent.image_decoder
            found = True
        else:
            decoder = deconvolution_factory()

        assert found or _is_external_ae, "Agent must have an image encoder or decoder!"

        model = nn.Sequential(
            encoder,
            Unflatten(),
            decoder,
        ).to(device)

        optimizer = build_optimizer(model.parameters(), learning_rate)

        total_items = 0
        data_type = 'any' if pretrain_CNNs_on_eval else 'train'


        for epoch_index in range(epochs):
            # Optimization
            epoch_loss, epoch_items = 0., 0. # TODO Do they need to be floats instead of integers?
            pbar = Progress.get_progress_cls(display_mode)(steps_per_epoch, epoch_index, logged_items={'L'})
            with pbar:
                for _ in range(steps_per_epoch):
                    optimizer.zero_grad()

                    batch = data_iterator.get_batch(data_type=data_type, keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[])
                    batch_img = batch.target_img(stack=True)
                    output = model(batch_img)

                    loss = F.mse_loss(output, batch_img, reduction="sum")

                    epoch_loss += loss.item()
                    epoch_items += batch_img.size(0)
                    total_items += batch_img.size(0)
                    if(self.autologger.summary_writer is not None): self.autologger.summary_writer.add_scalar(loss_tag, (loss.item() / batch_img.size(0)), total_items)
                    pbar.update(L=(epoch_loss / epoch_items))

                    loss.backward()
                    optimizer.step()

            # Evaluation
            with torch.no_grad():
                model.eval()
                test_epoch_loss, test_epoch_items = 0., 0.
                for _ in range(1 + (steps_per_epoch // 10)):
                    batch = data_iterator.get_batch(data_type='test', keep_category=True, no_evaluation=(not pretrain_CNNs_on_eval), sampling_strategies=[])
                    batch_img = batch.target_img(stack=True)
                    output = model(batch_img)
                    loss = F.mse_loss(output, batch_img, reduction="sum")
                    test_epoch_loss += loss.item()
                    test_epoch_items += batch_img.size(0)
                test_epoch_loss = test_epoch_loss / test_epoch_items
                if(self.autologger.summary_writer is not None):
                    self.autologger.summary_writer.writer.add_scalar('eval-' + loss_tag, test_epoch_loss, epoch_index)
                print(f"[eval-pretrain {agent_name} epoch {epoch_index}] L={test_epoch_loss}")
                model.train()

        return {'model': model}
