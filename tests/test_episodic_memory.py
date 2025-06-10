"""Test the EpisodicMemory module."""

from collections.abc import Generator

import pytest
import torch
from memory.episodic_memory import EpisodicMemory


@pytest.fixture
def episodic_memory_module() -> Generator[EpisodicMemory, None, None]:
    """Create and return a fresh instance of the EpisodicMemory module."""
    dim = 32
    memory_size = 50
    # The 'yield' keyword provides the object to the test.
    # Code after yield would be for teardown.
    yield EpisodicMemory(dim, memory_size)
    # Teardown: No teardown needed for this fixture.
    return None


# --- Pytest Test Function ---
def test_state_changes_during_inference(episodic_memory_module):
    """Verify that memory.update() modifies the internal state during eval mode.

    Note: The test function takes the fixture as an argument.
    """
    # 1. Setup: Fixture provides the module. Now set it to eval mode.
    memory = episodic_memory_module
    memory.eval()

    # 2. Snapshot the 'before' state of the internal memory
    keys_before_update = memory.mem_keys.clone().detach()

    # 3. Perform the update action within a no_grad context
    with torch.no_grad():
        batch_size = 4
        new_keys = torch.randn(batch_size, memory.mem_keys.shape[1])
        new_values = torch.randn(batch_size, memory.mem_values.shape[1])
        memory.update(new_keys, new_values)

    # 4. Assert using a standard Python `assert` statement
    keys_after_update = memory.mem_keys.detach()

    # Pytest uses the plain `assert` for expressive and detailed failure reports.
    assert not torch.allclose(
        keys_before_update, keys_after_update
    ), "The memory's internal keys should have been modified by the update() call."
