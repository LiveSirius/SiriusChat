"""向后兼容重定向 —— 实际实现已移至 sirius_chat.github.webhook。

新代码应使用::

    from sirius_chat.github import GitHubWebhookServer, verify_signature
"""

from sirius_chat.github.webhook import (
    GitHubWebhookServer,
    RepoFilter,
    WebhookHandler,
    verify_signature,
)

__all__ = [
    "GitHubWebhookServer",
    "RepoFilter",
    "WebhookHandler",
    "verify_signature",
]
