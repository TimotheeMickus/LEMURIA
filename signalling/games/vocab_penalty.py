"""Shared vocabulary-usage penalty for the signalling games.

Both games (image: Alice→Bob; predicate: Alex→Beth) drive the speaker with REINFORCE
and use the exact same symbol layout (the shared `SignalDecoder`), so the vocabulary
penalty is identical on both sides and lives here rather than being duplicated.

Two interchangeable mechanisms are offered, selected by `--voc_penalty_mode`:

* ``reward`` (the original): each symbol of the base alphabet carries a total penalty of
  ``voc_penalty * batch_size`` spread over its occurrences in the batch in inverse proportion
  to its use, subtracted from the speaker's REINFORCE reward. Summed over the batch this equals
  ``voc_penalty * batch_size * (number of distinct content symbols used)``, i.e. a flat L0-style
  charge on the size of the alphabet in use, delivered through the (high-variance) policy gradient.

* ``aux`` (a direct auxiliary loss): a differentiable group-sparsity penalty
  ``voc_penalty * sum_a m_a**p`` over the *content* symbols, where ``m_a = E_s[π(a|s)]`` is the
  batch symbol marginal and ``0 < p <= 1``. Because the marginal (near-)sums to a constant, a
  linear penalty (p = 1) is inert; a strictly concave one (p < 1) is Schur-concave and so pushes
  mass onto few symbols, shrinking the used alphabet (p → 0 approaches an L0 support count). Its
  gradient ``1/(p * m_a**(1-p))`` concentrates on already-small symbols, pruning the tail towards
  zero while barely disturbing the working vocabulary. Unlike a marginal-entropy penalty it shares
  no term with the speaker's conditional-entropy (beta) exploration bonus, so the two do not cancel
  term-for-term and the coefficient stays meaningfully tunable.

The two modes are on different scales; `voc_penalty` needs separate tuning when switching.
"""

import torch


class VocabularyPenaltyMixin:
    # Reads the three vocabulary-penalty knobs onto `self`. Call from each game's __init__.
    def _setup_vocab_penalty(self, args):
        self.voc_penalty = args.voc_penalty
        self.voc_penalty_mode = getattr(args, "voc_penalty_mode", "reward") # 'reward' (original) or 'aux' (group-sparsity auxiliary loss).
        self.voc_penalty_p = getattr(args, "voc_penalty_p", 0.5) # exponent for the 'aux' group-sparsity penalty.

    # 'reward' mode. Returns the per-signal penalty to *subtract* from the speaker reward:
    # either the float 0.0 (penalty disabled or a different mode is selected) or a tensor of shape (batch,).
    # signal_tokens: long tensor of shape (batch, max signal length), including EOS and padding.
    # eos_index, padding_idx: symbols exempt from the penalty. full_alphabet_size: bincount width.
    def vocabulary_reward_penalty(self, signal_tokens, eos_index, padding_idx, full_alphabet_size):
        if(not (self.voc_penalty > 0.0 and self.voc_penalty_mode == "reward")):
            return 0.0

        vocabulary_counts = torch.bincount(signal_tokens.view(-1), minlength=full_alphabet_size) # Shape: (full_alphabet_size,), includes padding and EOS

        vocabulary_freqs = 1.0 / vocabulary_counts # Shape: (full_alphabet_size,); inf for unused symbols (never gathered below)
        vocabulary_freqs[eos_index] = 0.0 # no penalty for using EOS
        vocabulary_freqs[padding_idx] = 0.0 # no penalty for "using" padding

        batch_size = signal_tokens.shape[0]
        symbol_penalty = (self.voc_penalty * batch_size) * vocabulary_freqs # Shape: (full_alphabet_size,)

        return symbol_penalty[signal_tokens].sum(dim=1) # Shape: (batch,)

    # 'aux' mode. Returns the scalar loss to *add* to the speaker loss: either the float 0.0
    # (penalty disabled or a different mode is selected) or a differentiable scalar tensor.
    # symbol_marginal: differentiable tensor of shape (base_alphabet_size + 1,) (index eos_index is EOS).
    def vocabulary_aux_loss(self, symbol_marginal, eos_index):
        if(not (self.voc_penalty > 0.0 and self.voc_penalty_mode == "aux")):
            return 0.0

        eps = 1e-9 # avoids the infinite subgradient of m**p at m = 0 for p < 1.
        content_mask = torch.ones_like(symbol_marginal, dtype=torch.bool)
        content_mask[eos_index] = False # exclude EOS; padding/BOS are not in the marginal's support.
        content_mass = symbol_marginal[content_mask].clamp_min(eps) # Shape: (base_alphabet_size,)

        return self.voc_penalty * (content_mass ** self.voc_penalty_p).sum()
