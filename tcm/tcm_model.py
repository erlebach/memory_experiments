# tcm_model.py
import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor


class TCM_A:
    """A minimal implementation of the Temporal Context Model with Accumulators (TCM-A).

    This model demonstrates contextual drift and retrieved context principles for
    free recall, based on the work of Sederberg, Howard, and Kahana (2008).[1, 1]
    """

    def __init__(self, item_count: int, feature_count: int, params: dict):
        """Initialize the TCM-A model.

        Args:
            item_count (int): The number of unique items in the word pool.
            feature_count (int): The dimensionality of item and context vectors.
            params (dict): A dictionary of model parameters.

        """
        self.item_count = item_count
        self.feature_count = feature_count
        self.params = params
        self.rng = np.random.default_rng()

        # Create orthogonal representations for all possible items [1]
        self.item_representations = np.eye(item_count, feature_count)

    def _initialize_memory(self, list_length: int) -> None:
        """Initialize memory stores for a new list."""
        # Pre-experimental matrices are treated as identity matrices for simplicity,
        # implying no pre-existing semantic associations.[1]
        self.M_pre_ft = np.eye(self.feature_count)

        # Experimental (newly learned) associations start at zero [1]
        self.M_exp_ft = np.zeros((self.feature_count, self.feature_count))
        self.M_exp_tf = np.zeros((self.feature_count, self.feature_count))

        # Context vector starts as a zero vector (dynamic vector)
        self.context = np.zeros(self.feature_count)
        self.recalled_items_indices = set()

    def _update_context(
        self,
        t_in: Float[NDArray, "feature_count"],  # noqa: F821
        beta: float,
    ) -> None:
        """Update the context vector according to the core TCM equation.

        t_i = rho * t_{i-1} + beta * t_in

        Where rho is calculated to maintain unit norm constraint ||t_i|| = 1.
        Based on equation (7) from Howard & Kahana (2002).

        """
        # Handle the case where context is initially zero
        if np.linalg.norm(self.context) == 0:
            # If context is zero, just use the input (normalized)
            self.context = (
                t_in / np.linalg.norm(t_in) if np.linalg.norm(t_in) > 0 else t_in
            )
            return

        # Calculate dot product between previous context and input
        dot_product = np.dot(self.context, t_in)

        # Calculate rho using equation (7) from Howard & Kahana (2002)
        # ρᵢ = √[1 + β²[(tᵢ₋₁ · tᵢᴵᴺ)² - 1]] - β(tᵢ₋₁ · tᵢᴵᴺ)
        discriminant = 1 + beta**2 * (dot_product**2 - 1)

        # Ensure discriminant is non-negative for real solution
        if discriminant < 0:
            discriminant = 0

        rho = np.sqrt(discriminant) - beta * dot_product

        # Update context vector
        self.context = rho * self.context + beta * t_in

        # Verify unit norm (should be automatically satisfied by the rho calculation)
        # but normalize as a safety check for numerical precision
        context_norm = np.linalg.norm(self.context)
        if context_norm > 0:
            self.context /= context_norm

    def encode_list(
        self, item_indices: list[int], distractor_duration: int = 0
    ) -> None:
        """Encode a list of items, updating context and associative matrices.

        Args:
            item_indices (list): A list of integer indices for the items to be encoded.
            distractor_duration (int): Number of distractor steps after the list.

        """
        self._initialize_memory(len(item_indices))

        for i, item_idx in enumerate(item_indices):
            # one-hot encode the item
            item_vec: NDArray = self.item_representations[item_idx]

            # 1. Form context-to-item association with pre-update context (t_{i-1}) [1]
            primacy_boost: float = (
                self.params["phi_s"] * np.exp(-self.params["phi_d"] * i) + 1
            )

            # NDArray,shape: 50, 50
            self.M_exp_tf += primacy_boost * np.outer(item_vec, self.context)

            # 2. Retrieve pre-experimental context to drive contextual drift [1]
            # (feature_count, feature_count) @ (feature_count,) -> (feature_count,)
            t_in: NDArray = self.M_pre_ft @ item_vec

            # 3. Update context vector
            self._update_context(t_in, self.params["beta_enc"])

        # Simulate post-list distractor period if applicable
        for _ in range(distractor_duration):
            distractor_vec = self.rng.normal(size=self.feature_count)
            distractor_vec /= np.linalg.norm(distractor_vec)
            self._update_context(distractor_vec, self.params["beta_dist"])

    def recall(
        self, list_length: int, time_limit: int = 1000
    ) -> tuple[list[int], list[int]]:
        """Simulate free recall with timing information.

        Args:
            list_length: The length of the list being recalled.
            time_limit: Maximum number of time steps for the recall process.

        Returns:
            tuple: (recalled_items, recall_times) where recall_times[i] is the
                   time step when recalled_items[i] was recalled.

        """
        recalled_sequence = []
        recall_times = []

        # Initialize accumulators for all items in the pool
        accumulators = np.zeros(self.item_count)

        for time_step in range(time_limit):
            # 1. Calculate item activations from the current context cue [1]
            # This combines activation from newly learned and pre-experimental associations
            item_activations = (
                self.context @ (self.params["gamma_tf"] * self.M_exp_tf).T
            )

            # 2. Update accumulators based on the Usher & McClelland (2001) model [1]
            noise = self.params["sigma"] * self.rng.normal(self.item_count)

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
                    recall_times.append(time_step)  # Record when item was recalled
                    self.recalled_items_indices.add(winner_idx)

                    # Check if we've recalled the expected number of items
                    if len(recalled_sequence) >= list_length:
                        break  # Successful complete recall

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

        return recalled_sequence, recall_times

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
