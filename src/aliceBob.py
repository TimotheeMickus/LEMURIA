import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import tqdm

from sender import Sender
from receiver import Receiver
from senderReceiver import SenderReceiver
import utils
from utils import Progress, show_imgs, max_normalize_, to_color, pointing, add_normal_noise

class AliceBob(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        shared = args.shared

        if(shared):
            senderReceiver = SenderReceiver.from_args(args)
            self.sender = senderReceiver.sender
            self.receiver = senderReceiver.receiver
        else:
            self.sender = Sender.from_args(args)
            self.receiver = Receiver.from_args(args)

        self.use_expectation = args.use_expectation
        self.grad_scaling = args.grad_scaling or 0
        self.grad_clipping = args.grad_clipping or 0
        self.beta_sender = args.beta_sender
        self.beta_receiver = args.beta_receiver
        self.penalty = args.penalty
        self.adaptative_penalty = args.adaptative_penalty


    def _bob_input(self, batch):
        return torch.cat([batch.target_img.unsqueeze(1), batch.base_distractors], dim=1)

    def _forward(self, batch, sender, receiver):
        sender_outcome = sender(batch.original_img)
        receiver_outcome = receiver(self._bob_input(batch), *sender_outcome.action)

        return sender_outcome, receiver_outcome

    def forward(self, batch):
        """
        Input:
            `batch` is a Batch (a kind of named tuple); 'original_img' and 'target_img' are tensors of shape [args.batch_size, *IMG_SHAPE] and 'base_distractors' is a tensor of shape [args.batch_size, 2, *IMG_SHAPE]
        Output:
            `sender_outcome`, sender.Outcome
            `receiver_outcome`, receiver.Outcome
        """
        return self._forward(batch, self.sender, self.receiver)

    def decision_tree(self, data_iterator):
        base_alphabet_size = self.get_base_alphabet_size()
        self.eval()

        print("Generating the messages…")
        messages = []
        with torch.no_grad():
            for datapoint in tqdm.tqdm(data_iterator.dataset):
                sender_outcome = self.sender(datapoint.img.unsqueeze(0))
                message = sender_outcome.action[0].view(-1).tolist()
                messages.append(message)
                #print((datapoint.category, message))

        # As features, we will use the presence of n-grams
        import numpy as np

        n = 3
        alphabet_size = base_alphabet_size + 1
        nb_ngrams = alphabet_size * (alphabet_size**n - 1) // (alphabet_size - 1)
        print('Number of possible %i-grams: %i' % (n, nb_ngrams))

        ngrams = [()] * nb_ngrams
        def ngram_to_idx(ngram): # `ngram` is a list of integers
            idx = 0
            for i, k in enumerate(ngram): # We read the n-grams as numbers in base ALPHABET_SIZE written in reversed and with '0' used as the unit, instead of '1' (because message (0, 0) is different from (0))
                idx += (k + 1) * (alphabet_size**i) # '+1' because symbol '0' is used as the unit

            idx -= 1 # Because the 0-gram is not taken into account

            assert (ngrams[idx] == () or ngrams[idx] == ngram) # Checks that we are not assigning the same id to two different n-grams

            ngrams[idx] = ngram

            return idx

        last_symbol = alphabet_size - 1 # Because the alphabet starts with 0
        last_tuple = tuple([last_symbol] * n)
        print('Id of %s: %i' % (last_tuple, ngram_to_idx(last_tuple)))

        feature_vectors = []
        for message in messages:
            # We could consider adding the BOM symbol
            v = np.zeros(nb_ngrams, dtype=bool)
            s = set()
            for l in range(1, (n + 1)):
                for i in range(len(message) - l + 1):
                    ngram = tuple(message[i:(i + l)])
                    s.add(ngram)
                    idx = ngram_to_idx(ngram)
                    v[idx] = True
                    #print((ngram, idx))
            #input((message, v, s))
            feature_vectors.append(v)

        feature_vectors = np.array(feature_vectors)
       
        import sklearn.tree
        import matplotlib.pyplot as plt
        import itertools

        results_decision_tree = []
        max_depth = 2 # None # We could successively try with increasing depth
        max_conjunctions = 3 # data_iterator.nb_concepts
        for size_conjunctions in range(1, (max_conjunctions + 1)):
            results_binary_classifier = []

            for concept_indices in itertools.combinations(range(data_iterator.nb_concepts), size_conjunctions): # Iterates over all subsets of [|0, `data_iterator.nb_concepts`|[ of size `size_conjunctions`
                #print([data_iterator.concept_names[idx] for idx in concept_indices])

                # For each selected concept, we pick a value
                conjunctions = itertools.product(*[data_iterator._concepts[idx].keys() for idx in concept_indices])

                for conjunction in conjunctions:
                    #print('\t class: %s' % str(conjunction))

                    def in_class(category):
                        for i, idx in enumerate(concept_indices):
                            if(category[idx] != data_iterator._concepts[idx][conjunction[i]]): return False

                        return True

                    in_class_aux = np.vectorize(lambda datapoint: in_class(datapoint.category))

                    # For each n-gram, check if it is a good predictor of the class (equivalent to building a decision tree of depth 1)
                    gold = in_class_aux(data_iterator.dataset)
                    for feature_idx in range(nb_ngrams):
                        if(ngrams[feature_idx] == ()): continue

                        ratio = gold.mean()
                        baseline_accuracy = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                        feature_type = 'presence'
                        prediction = feature_vectors[:, feature_idx]

                        matches = (gold == prediction)

                        accuracy = matches.mean()
                        error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                        precision = gold[prediction].mean()
                        recall = prediction[gold].mean()
                        f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                        item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, ngrams[feature_idx], feature_type)
                        results_binary_classifier.append(item)

                        feature_type = 'absence'
                        prediction = (prediction ^ True)

                        matches = (gold == prediction)

                        accuracy = matches.mean()
                        error_reduction = (1 - baseline_accuracy) / (1 - accuracy)

                        precision = gold[prediction].mean()
                        recall = prediction[gold].mean()
                        f1 = (2 * precision * recall / (precision + recall)) if(precision + recall > 0.0) else 0.0

                        item = (accuracy, baseline_accuracy, error_reduction, precision, recall, f1, conjunction, ngrams[feature_idx], feature_type)
                        results_binary_classifier.append(item)


                    if(True): continue

                    # Decision trees
                    X = feature_vectors
                    Y = gold # in_class_aux(data_iterator.dataset) # [in_class(datapoint.category) for datapoint in data_iterator.dataset]

                    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth).fit(X, Y)

                    n_leaves, depth = classifier.get_n_leaves(), classifier.get_depth()
                    precision = classifier.score(X, Y) # Precision on the 'training set'
                    ratio = (np.sum(Y) / Y.size)
                    baseline_precision = max(ratio, (1.0 - ratio)) # Precision of the majority class baseline

                    item = (
                        precision,
                        baseline_precision,
                        (precision / baseline_precision),
                        ((1 - baseline_precision) / (1 - precision)),
                        conjunction,
                        n_leaves,
                        depth,
                        classifier
                    )

                    results_decision_tree.append(item)

                    #if(precision > 0.9):
                    if(item[3] > 2.0):
                        print(item)
                        print(sklearn.tree.export_text(classifier, feature_names=ngrams, show_weights=True))

                        plt.figure(figsize=(12, 12))
                        sklearn.tree.plot_tree(classifier, filled=True)
                        plt.show()

            print("\nBest binary classifiers")
            print("\tby error reduction")
            results_binary_classifier.sort(reverse=True, key=(lambda e: e[2]))
            for e in results_binary_classifier[:10]:
                print(e)

            print("\tby F1")
            results_binary_classifier.sort(reverse=True, key=(lambda e: e[5]))
            for e in results_binary_classifier[:10]:
                print(e)

        print("\nBest decision trees")
        results_decision_tree.sort(reverse=True, key=(lambda e: e[3]))
        for e in results_decision_tree[:10]:
            print(e)

    def test_visualize(self, data_iterator, learning_rate):
        self.eval() # Sets the model in evaluation mode; good idea or not?

        batch_size = 4
        batch = data_iterator.get_batch(batch_size)

        batch.original_img.requires_grad = True
        batch.target_img.requires_grad = True
        batch.base_distractors.requires_grad = True

        sender_outcome, receiver_outcome = self(batch)

        # Image-specific saliency visualisation (inspired by Simonyan et al. 2013)
        pseudo_optimizer = torch.optim.Optimizer([batch.original_img, batch.target_img, batch.base_distractors], {}) # I'm defining this only for its `zero_grad` method (but maybe we won't need it)
        pseudo_optimizer.zero_grad()

        _COLOR, _INTENSITY = range(2)
        def process(t, dim, mode):
            if(mode == _COLOR):
                t = max_normalize(t, dim=dim, abs_val=True) # Normalises each image
                t *= 0.5
                t += 0.5

                return t
            elif(mode == _INTENSITY):
                t = t.abs()
                t = t.max(dim).values # Max over the colour channel
                max_normalize_(t, dim=dim, abs_val=False) # Normalises each image
                return to_color(t, dim)

        mode = _INTENSITY

        # Alice's part
        sender_outcome.log_prob.sum().backward()

        sender_part = batch.original_img.grad.detach()
        sender_part = process(sender_part, 1, mode)

        # Bob's part
        receiver_outcome.scores.sum().backward()

        receiver_part_target_img = batch.target_img.grad.detach()
        receiver_part_target_img = process(receiver_part_target_img.unsqueeze(axis=1), 2, mode).squeeze(axis=1)

        receiver_part_base_distractors = batch.base_distractors.grad.detach()
        receiver_part_base_distractors = process(receiver_part_base_distractors, 2, mode)

        # Message Bob-model visualisation (inspired by Simonyan et al. 2013)
        #receiver_dream = add_normal_noise((0.5 + torch.zeros_like(batch.original_img)), std_dev=0.1, clamp_values=(0,1)) # Starts with normally-random images
        receiver_dream = torch.stack([data_iterator.average_image() for _ in range(batch_size)]) # Starts with the average of the dataset
        #show_imgs([data_iterator.average_image()], 1)
        receiver_dream = receiver_dream.unsqueeze(axis=1) # Because the receiver expect a 1D array of images per batch instance; shape: [batch_size, 1, 3, height, width]
        receiver_dream.requires_grad = True

        encoded_message = self.receiver.encode_message(*sender_outcome.action).detach()

        # Defines a filter for checking smoothness
        channels = 3
        filter_weight = torch.tensor([[1.2, 2, 1.2], [2, -12.8, 2], [1.2, 2, 1.2]]) # -12.8 (at the center) is equal to the opposite of sum of the other coefficient
        filter_weight = filter_weight.view(1, 1, 3, 3)
        filter_weight = filter_weight.repeat(channels, 1, 1, 1) # Shape: [channel, 1, 3, 3]
        filter_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, groups=channels, bias=False)
        filter_layer.weight.data = filter_weight
        filter_layer.weight.requires_grad = False

        #optimizer = torch.optim.RMSprop([receiver_dream], lr=10.0*args.learning_rate)
        optimizer = torch.optim.SGD([receiver_dream], lr=2*learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam([receiver_dream], lr=10.0*args.learning_rate)
        nb_iter = 1000
        j = 0
        for i in range(nb_iter):
            if(i >= (j + (nb_iter / 10))):
                print(i)
                j = i

            optimizer.zero_grad()

            tmp_outcome = self.receiver.aux_forward(receiver_dream, encoded_message)
            loss = -tmp_outcome.scores[:, 0].sum()

            regularisation_loss = 0.0
            #regularisation_loss += 0.05 * (receiver_dream - 0.5).norm(2) # Similar to L2 regularisation but centered around 0.5
            regularisation_loss += 0.01 * (receiver_dream - 0.5).norm(1) # Similar to L1 regularisation but centered around 0.5
            #regularisation_loss += -0.1 * torch.log(1.0 - (2 * torch.abs(receiver_dream - 0.5))).sum() # "Wall" at 0 and 1
            loss += regularisation_loss

            #smoothness_loss = 20 * torch.abs(filter_layer(receiver_dream.squeeze(axis=1))).sum()
            smoothness_loss = 20 * torch.abs(filter_layer(receiver_dream.squeeze(axis=1))).norm(1)
            loss += smoothness_loss

            loss.backward()

            optimizer.step()
        receiver_dream = receiver_dream.squeeze(axis=1)
        receiver_dream = torch.clamp(receiver_dream, 0, 1)

        # Displays the visualisations
        imgs = []
        for i in range(batch_size):
            imgs.append(batch.original_img[i].detach())
            imgs.append(sender_part[i])

            imgs.append(batch.target_img[i].detach())
            imgs.append(receiver_part_target_img[i])

            for j in range(batch.base_distractors.size(1)):
                imgs.append(batch.base_distractors[i][j].detach())
                imgs.append(receiver_part_base_distractors[i][j])

            imgs.append(receiver_dream[i].detach())
        #for img in imgs: print(img.shape)
        show_imgs(imgs, nrow=(len(imgs) // batch_size)) #show_imgs(imgs, nrow=(2 * (2 + batch.base_distractors.size(1))))

    def compute_sender_rewards(self, sender_action, receiver_scores, running_avg_success):
        """
            returns the reward as well as the success for each element of a batch
        """
        # Generates a probability distribution from the scores and sample an action
        receiver_pointing = pointing(receiver_scores)

        # By design, the target is the first image
        if(self.use_expectation): successes = receiver_pointing['dist'].probs[:, 0].detach()
        else: successes = (receiver_pointing['action'] == 0).float() # Plays dice

        rewards = successes

        if(self.penalty > 0.0):
            msg_lengths = sender_action[1].view(-1).float() # Float casting could be avoided if we upgrade torch to 1.3.1; see https://github.com/pytorch/pytorch/issues/9515 (I believe)
            length_penalties = 1.0 - (1.0 / (1.0 + self.penalty * msg_lengths)) # Equal to 0 when `args.penalty` is set to 0, increases to 1 with the length of the message otherwise

            # TODO J'ai peur que ce système soit un peu trop basique et qu'il encourage le système à être sous-performant - qu'on puisse obtenir plus de reward en faisant exprès de se tromper.
            if(self.adaptative_penalty):
                chance_perf = (1 / receiver_scores.size(1))
                improvement_factor = (running_avg_success - chance_perf) / (1 - chance_perf) # Equals 0 when running average equals chance performance, reaches 1 when running average reaches 1
                length_penalties = (length_penalties * min(0.0, improvement_factor))

            rewards = (rewards - length_penalties)

        return (rewards, successes)

    def compute_sender_loss(self, sender_outcome, receiver_scores, running_avg_success):
        (rewards, successes) = self.compute_sender_rewards(sender_outcome.action, receiver_scores, running_avg_success)
        log_prob = sender_outcome.log_prob.sum(dim=1)

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_sender * sender_outcome.entropy.mean()) # Entropy penalty

        return (loss, successes, rewards)

    def compute_receiver_loss(self, receiver_scores):
        receiver_pointing = pointing(receiver_scores) # The sampled action is not the same as the one in `sender_rewards` but it probably does not matter

        # By design, the target is the first image
        if(self.use_expectation):
            successes = receiver_pointing['dist'].probs[:, 0].detach()
            log_prob = receiver_pointing['dist'].log_prob(torch.tensor(0).to(probs.device))
        else: # Plays dice
            successes = (receiver_pointing['action'] == 0).float()
            log_prob = receiver_pointing['dist'].log_prob(receiver_pointing['action'])

        rewards = successes

        loss = -(rewards * log_prob).mean()

        loss = loss - (self.beta_receiver * receiver_pointing['dist'].entropy().mean()) # Entropy penalty

        return loss

    def get_base_alphabet_size(self):
        if hasattr(self, 'sender'):
            return self.sender.message_decoder.symbol_embeddings.weight.size(0) - 1
        elif hasattr(self, 'senders'):
            return self.senders[0].message_decoder.symbol_embeddings.weight.size(0) - 1
        else:
            raise TypeError

    def train_epoch(self, data_iterator, optim, epoch=1, steps_per_epoch=1000, event_writer=None, simple_display=False, debug=False, log_lang_progress=True, log_entropy=False):
        """
            Model training function
            Input:
                `data_iterator`, an infinite iterator over (batched) data
                `optim`, the optimizer
            Optional arguments:
                `epoch`: epoch number to display in progressbar
                `steps_per_epoch`: number of steps for epoch
                `event_writer`: tensorboard writer to log evolution of values
        """
        self.train() # Sets the model in training mode

        base_alphabet_size = self.get_base_alphabet_size()

        with Progress(simple_display, steps_per_epoch, epoch) as pbar:
            total_reward = 0.0 # sum of the rewards since the beginning of the epoch
            total_success = 0.0 # sum of the successes since the beginning of the epoch
            total_items = 0 # number of training instances since the beginning of the epoch
            running_avg_reward = 0.0
            running_avg_success = 0.0
            start_i = ((epoch - 1) * steps_per_epoch) + 1 # (the first epoch is numbered 1, and the first iteration too)
            end_i = start_i + steps_per_epoch
            device = next(self.parameters()).device
            past_dist, current_dist = None, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device) # size of embeddings

            if event_writer is not None and log_entropy:
                symbol_counts = torch.zeros(base_alphabet_size, dtype=torch.float).to(device)
            for i, batch in zip(range(start_i, end_i), data_iterator):
                optim.zero_grad()
                sender_outcome, receiver_outcome = self(batch)

                # Alice's part
                (sender_loss, sender_successes, sender_rewards) = self.compute_sender_loss(sender_outcome, receiver_outcome.scores, running_avg_success)

                # Bob's part
                receiver_loss = self.compute_receiver_loss(receiver_outcome.scores)

                loss = sender_loss + receiver_loss

                loss.backward() # Backpropagation

                # Gradient clipping and scaling
                if(self.grad_clipping > 0): torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clipping)
                if(self.grad_scaling > 0): torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_scaling)

                optim.step()

                rewards = sender_rewards
                successes = sender_successes

                avg_reward = rewards.mean().item() # average reward of the batch
                avg_success = successes.mean().item() # average success of the batch
                avg_msg_length = sender_outcome.action[1].float().mean().item() # average message length of the batch

                # updates running average reward
                total_reward += rewards.sum().item()
                total_success += successes.sum().item()
                total_items += batch.size
                running_avg_reward = total_reward / total_items
                running_avg_success = total_success / total_items

                if log_lang_progress:
                    batch_msg_manyhot = torch.zeros((batch.size, base_alphabet_size + 2), dtype=torch.float).to(device) # size of embeddings + EOS + PAD
                    # message -> many-hot
                    many_hots = batch_msg_manyhot.scatter_(1,sender_outcome.action[0].detach(),1).narrow(1,1,base_alphabet_size).float()
                    # summation along batch dimension,  and add to counts
                    current_dist += torch.einsum('bi,bj->ij', many_hots, batch.original_category.float().to(device)).detach().float()

                pbar.update(R=running_avg_success)

                # logs some values
                if(event_writer is not None):
                    number_ex_seen = i * batch.size
                    event_writer.add_scalar('train/reward', avg_reward, number_ex_seen)
                    event_writer.add_scalar('train/success', avg_success, number_ex_seen)
                    event_writer.add_scalar('train/loss', loss.item(), number_ex_seen)
                    event_writer.add_scalar('llp/msg_length', avg_msg_length, number_ex_seen)
                    if debug:
                        median_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().median().item()
                        mean_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().mean().item()
                        max_grad = torch.cat([p.grad.view(-1).detach() for p in self.parameters()]).abs().max().item()
                        mean_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).mean().item()
                        max_norm_grad = torch.stack([p.grad.view(-1).detach().data.norm(2.) for p in self.parameters()]).max().item()
                        event_writer.add_scalar('grad/median_grad', median_grad, number_ex_seen)
                        event_writer.add_scalar('grad/mean_grad', mean_grad, number_ex_seen)
                        event_writer.add_scalar('grad/max_grad', max_grad, number_ex_seen)
                        event_writer.add_scalar('grad/mean_norm_grad', mean_norm_grad, number_ex_seen)
                        event_writer.add_scalar('grad/max_norm_grad', max_norm_grad, number_ex_seen)

                    if log_lang_progress and i%100 == 0:
                        if past_dist is None:
                            past_dist, current_dist = current_dist, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device)
                            continue
                        else:
                            logit_c = (current_dist.view(1, -1) / current_dist.sum()).log()
                            prev_p = (past_dist.view(1, -1) / past_dist.sum())
                            kl = F.kl_div(logit_c, prev_p, reduction='batchmean').item()
                            event_writer.writer.add_scalar('llp/kl_div', kl, number_ex_seen)
                            past_dist, current_dist = current_dist, torch.zeros((base_alphabet_size, 5), dtype=torch.float).to(device)
                    if log_entropy:
                        new_messages = sender_outcome.action[0].view(-1)
                        valid_indices = torch.arange(base_alphabet_size).expand(new_messages.size(0), base_alphabet_size).to(device)
                        selected_symbols = valid_indices == new_messages.unsqueeze(1).float()
                        symbol_counts += selected_symbols.sum(dim=0)

        if log_entropy and (event_writer is not None):
            event_writer.writer.add_scalar('llp/entropy', utils.compute_entropy(symbol_counts), number_ex_seen)




        self.eval()
