from typing import Any

import numpy as np
from modified.rb import ReplayBuffer_custom
from modified.manager import ReplayBufferManager_custom

from tianshou.data import (
    PrioritizedReplayBuffer,
    PrioritizedReplayBufferManager,
    ReplayBuffer,
    ReplayBufferManager,
)


class VectorReplayBuffer_custom(ReplayBufferManager_custom):
    """VectorReplayBuffer contains n ReplayBuffer with the same size.
    It is used for storing transition from different environments yet keeping the order
    of time.
    :param int total_size: the total size of VectorReplayBuffer.
    :param int buffer_num: the number of ReplayBuffer it uses, which are under the same
        configuration.
    Other input arguments (stack_num/ignore_obs_next/save_only_last_obs/sample_avail)
    are the same as :class:`~tianshou.data.ReplayBuffer`.
    .. seealso::
        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [ReplayBuffer_custom(size, **kwargs) for _ in range(buffer_num)]
        super().__init__(buffer_list)
