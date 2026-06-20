from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

try:
    import jax.numpy as jnp
    _JaxArray = jnp.ndarray
except ModuleNotFoundError:
    _JaxArray = Any

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, torch.Tensor, Any]  # Any covers jnp.ndarray when jax is unavailable
