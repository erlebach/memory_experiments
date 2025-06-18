# tcm_model.py
import numpy as np


class TCM_A:
    """A minimal implementation of the Temporal Context Model with Accumulators (TCM-A).

    This model demonstrates contextual drift and retrieved context principles for
    free recall, based on the work of Sederberg, Howard, and Kahana (2008).[1, 1]
    """

    def __init__(self, item_count, feature_count, params):
        """Initialize the TCM-A model.

        Args:
            item_count (int): The number of unique items in the word pool.
            feature_count (int): The dimensionality of item and context vectors.
            params (dict): A dictionary of model parameters.

        """
        self.item_count = item_count
        self.feature_count = feature_count
        self.params = params

        # Create orthogonal representations for all possible items [1]
        self.item_representations = np.eye(item_count, feature_count)

    def _initialize_memory(self, list_length) -> None:
        """Initialize memory stores for a new list."""
        # Pre-experimental matrices are treated as identity matrices for simplicity,
        # implying no pre-existing semantic associations.[1]
        self.M_pre_ft = np.eye(self.feature_count)

        # Experimental (newly learned) associations start at zero [1]
        self.M_exp_ft = np.zeros((self.feature_count, self.feature_count))
        self.M_exp_tf = np.zeros((self.feature_count, self.feature_count))

        # Context vector starts as a zero vector
        self.context = np.zeros(self.feature_count)
        self.recalled_items_indices = set()

    def _update_context(self, t_in, beta) -> None:
        """Update the context vector according to the core TCM equation [1].

        t_i = rho * t_{i-1} + beta * t_in
        """
        rho = np.sqrt(1 - beta**2)
        self.context = rho * self.context + beta * t_in
        # Normalize to maintain unit length
        if np.linalg.norm(self.context) > 0:
            self.context /= np.linalg.norm(self.context)

    def encode_list(self, item_indices, distractor_duration=0) -> None:
        """Encode a list of items, updating context and associative matrices.

        Args:
            item_indices (list): A list of integer indices for the items to be encoded.
            distractor_duration (int): Number of distractor steps after the list.

        """
        self._initialize_memory(len(item_indices))

        for i, item_idx in enumerate(item_indices):
            item_vec = self.item_representations[item_idx]

            # 1. Form context-to-item association with pre-update context (t_{i-1}) [1]
            primacy_boost = self.params["phi_s"] * np.exp(-self.params["phi_d"] * i) + 1
            self.M_exp_tf += primacy_boost * np.outer(item_vec, self.context)

            # 2. Retrieve pre-experimental context to drive contextual drift [1]
            t_in = self.M_pre_ft @ item_vec

            # 3. Update context vector
            self._update_context(t_in, self.params["beta_enc"])

        # Simulate post-list distractor period if applicable
        for _ in range(distractor_duration):
            distractor_vec = np.random.randn(self.feature_count)
            distractor_vec /= np.linalg.norm(distractor_vec)
            self._update_context(distractor_vec, self.params["beta_dist"])

    def recall(self, list_length: int, time_limit: int = 1000) -> list[int]:
        """Simulate the free recall process using a leaky, competitive accumulator.

        Args:
            list_length: The length of the list being recalled.
            time_limit: Maximum number of time steps for the recall process.

        Returns:
            A list of recalled item indices in order of recall.

        """
        recalled_sequence = []

        # Initialize accumulators for all items in the pool
        accumulators = np.zeros(self.item_count)

        for _ in range(time_limit):
            # 1. Calculate item activations from the current context cue [1]
            # This combines activation from newly learned and pre-experimental associations
            item_activations = (
                self.context @ (self.params["gamma_tf"] * self.M_exp_tf).T
            )

            # 2. Update accumulators based on the Usher & McClelland (2001) model [1]
            noise = self.params["sigma"] * np.random.randn(self.item_count)

            # Lateral inhibition term (simplified)
            inhibition = self.params["lambda"] * (np.sum(accumulators) - accumulators)

            # Accumulator update equation
            delta_x = (
                self.params["tau"]
                * (item_activations - self.params["kappa"] * accumulators - inhibition)
                + noise
            )
            accumulators += delta_x

            # Ensure accumulators don't go below zero
            accumulators[accumulators < 0] = 0

            # 3. Check for recall (threshold crossing)
            if np.any(accumulators > self.params["threshold"]):
                winner_idx = np.argmax(accumulators)

                if winner_idx not in self.recalled_items_indices:
                    recalled_sequence.append(winner_idx)
                    self.recalled_items_indices.add(winner_idx)

                    # 4. Update context with retrieved item's context (retrieved context mechanism) [1, 1]
                    winner_vec = self.item_representations[winner_idx]

                    # Combine pre-experimental and experimental context
                    retrieved_context_exp = self.M_exp_tf @ winner_vec
                    retrieved_context_pre = self.M_pre_ft @ winner_vec

                    t_in_retrieval = (
                        self.params["gamma_ft"] * retrieved_context_exp
                        + (1 - self.params["gamma_ft"]) * retrieved_context_pre
                    )

                    self._update_context(t_in_retrieval, self.params["beta_rec"])

                    # Reset all accumulators for the next recall competition
                    accumulators.fill(0)
                else:
                    # If already recalled, suppress this item for a while
                    accumulators[winner_idx] = 0

        return recalled_sequence

    def recognize(self, item_idx: int) -> tuple[float, bool]:
        """Simulate a simple recognition judgment.

        Args:
            item_idx: The index of the item to be recognized.

        Returns:
            A tuple containing (activation_strength, is_recognized).

        """
        # Get the item representation
        item_vec = self.item_representations[item_idx]

        # Calculate activation strength from current context
        # This combines both experimental and pre-experimental associations
        activation_exp = np.dot(self.context, self.M_exp_tf @ item_vec)
        activation_pre = np.dot(self.context, self.M_pre_ft @ item_vec)

        # Combine activations with weighting parameter
        total_activation = (
            self.params["gamma_tf"] * activation_exp
            + (1 - self.params["gamma_tf"]) * activation_pre
        )

        # Recognition decision based on threshold
        is_recognized = total_activation > self.params["recog_threshold"]

        return total_activation, is_recognized
