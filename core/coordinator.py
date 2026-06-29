"""core.coordinator 兼容垫片。

历史上部分模块/测试通过 ``core.coordinator`` 引用协调器，而实现位于
``distributed.coordinator``。此模块统一从分布式协调器重新导出，避免重复实现。
"""

from distributed.coordinator import (  # noqa: F401
    DistributedCoordinator,
    Coordinator,
    coordinator,
)

__all__ = ["DistributedCoordinator", "Coordinator", "coordinator"]
