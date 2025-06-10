import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    def __init__(self, dim, memory_size=100, update_steps=1, lr=0.01):
        super().__init__()
        # Use nn.Parameter for internal state that needs to be updated via backprop
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim))
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim))
        self.update_steps = update_steps

        # The module manages its own optimizer for test-time updates
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, query):
        # Standard retrieval logic
        attn_scores = torch.matmul(query, self.mem_keys.T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        retrieved_value = torch.matmul(attn_probs, self.mem_values)
        return retrieved_value

    def update(self, new_keys, new_values):
        # The key logic: temporarily enable gradients for this update step,
        # even if the parent model is in eval mode and under a no_grad() context.
        with torch.enable_grad():
            for _ in range(self.update_steps):
                self.optimizer.zero_grad()

                # Use a loss function to drive the update
                retrieved = self.forward(new_keys)
                loss = F.mse_loss(retrieved, new_values)
                loss.backward()  # Gradients are computed only for self.keys and self.values
                self.optimizer.step()
