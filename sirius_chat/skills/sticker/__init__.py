"""表情包 RAG 系统：自动学习、人格化检索、动态偏好调整。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sirius_chat.skills.sticker.models import StickerRecord, StickerPreference
from sirius_chat.skills.sticker.vector_store import StickerVectorStore
from sirius_chat.skills.sticker.indexer import StickerIndexer
from sirius_chat.skills.sticker.preference import StickerPreferenceManager
from sirius_chat.skills.sticker.learner import StickerLearner
from sirius_chat.skills.sticker.feedback import StickerFeedbackObserver

__all__ = [
    "StickerRecord",
    "StickerPreference",
    "StickerVectorStore",
    "StickerIndexer",
    "StickerPreferenceManager",
    "StickerLearner",
    "StickerFeedbackObserver",
    "init_sticker_system",
]


def init_sticker_system(
    work_path: Path | str,
    persona_name: str,
    provider_async: Any | None = None,
    basic_memory: Any | None = None,
    model_name: str = "gpt-4o-mini",
    token_callback: Any | None = None,
) -> dict[str, Any]:
    """初始化表情包系统。

    在 Engine 初始化时调用，创建所有必要的组件。

    Args:
        work_path: 工作目录
        persona_name: 人格名称
        provider_async: LLM provider
        basic_memory: 基础记忆管理器（用于反馈观察）
        model_name: 标签提取和偏好生成使用的模型

    Returns:
        包含所有组件的字典
    """
    sticker_work_path = Path(work_path) / "stickers"
    sticker_work_path.mkdir(parents=True, exist_ok=True)

    indexer = StickerIndexer(
        work_path=sticker_work_path,
        persona_name=persona_name,
    )
    indexer.load_from_disk()

    preference_manager = StickerPreferenceManager(
        work_path=sticker_work_path,
        persona_name=persona_name,
        model_name=model_name,
        token_callback=token_callback,
    )

    learner = StickerLearner(
        indexer=indexer,
        provider_async=provider_async,
        model_name=model_name,
        token_callback=token_callback,
    )

    feedback_observer = StickerFeedbackObserver(
        indexer=indexer,
        preference_manager=preference_manager,
        basic_memory=basic_memory,
    )

    return {
        "work_path": str(sticker_work_path),
        "indexer": indexer,
        "preference_manager": preference_manager,
        "learner": learner,
        "feedback_observer": feedback_observer,
    }
