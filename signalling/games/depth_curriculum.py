# Predicate-depth training curriculum for AlexBeth.
#
# When a threshold is given, training and evaluation start restricted to the shallowest predicates (structural depth == the dataset's minimum depth); each time the evaluation accuracy reaches the threshold, the next depth is unlocked. 
# The restriction is applied through the dataset's predicate-sampling mask, which both training and evaluation batches honour, so a single object drives both.
# All of the curriculum's mutable state lives here (rather than as fields on the game).
class DepthCurriculum:
    # threshold: float in [0, 1] or None (None disables the curriculum).
    # dataset: the predicate Dataset (must expose `predicate_depths` and `use_predicates_up_to_depth`).
    def __init__(self, threshold, dataset):
        self._threshold = threshold
        self._dataset = dataset

        depths = dataset.predicate_depths
        self._min_depth = int(depths.min())
        self._max_depth = int(depths.max())
        self._current_max_depth = self._min_depth
        self._last_unlock_epoch = None

        if(self.enabled):
            # Restriction to the shallowest predicates.
            self._dataset.use_predicates_up_to_depth(self._current_max_depth)

    @property
    def enabled(self):
        return (self._threshold is not None)

    @property
    def threshold(self):
        return self._threshold

    @property
    def min_depth(self):
        return self._min_depth

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def current_max_depth(self):
        return self._current_max_depth

    # If enabled and the evaluation accuracy clears the threshold, unlock the next depth (a single level) and re-apply the dataset mask.
    # Returns True iff a depth was just unlocked.
    def maybe_unlock(self, eval_accuracy, epoch_index):
        if(not self.enabled):
            return False
        if(self._current_max_depth >= self._max_depth):
            return False
        if(eval_accuracy < self._threshold):
            return False

        self._current_max_depth += 1
        self._dataset.use_predicates_up_to_depth(self._current_max_depth)
        self._last_unlock_epoch = epoch_index
        return True

    # True on the evaluation immediately following a depth unlock (i.e. the predicate set has just been enlarged); used to reset the performance baseline and to force a signal dump.
    def is_epoch_after_unlock(self, epoch_index):
        return (self._last_unlock_epoch is not None) and (epoch_index == (self._last_unlock_epoch + 1))

    # One-line status suitable for the terminal.
    def status_line(self, epoch_index):
        return (f"[depth-curriculum] epoch {epoch_index}: current_max_depth={self._current_max_depth} "
                f"(min {self._min_depth}, cap {self._max_depth})")
