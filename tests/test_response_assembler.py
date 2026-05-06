"""Tests for ResponseAssembler and StyleAdapter (v0.28 execution layer)."""

from __future__ import annotations

import pytest

from sirius_chat.core.response_assembler import ResponseAssembler, StyleAdapter, StyleParams
from sirius_chat.models.emotion import EmotionState
from sirius_chat.models.intent_v3 import IntentAnalysisV3, SocialIntent
from sirius_chat.models.models import Message
from sirius_chat.memory.semantic.models import (
    GroupSemanticProfile,
    UserSemanticProfile,
    AtmosphereSnapshot,
)


class TestStyleAdapter:
    def test_hot_accelerating_limits_tokens(self):
        adapter = StyleAdapter()
        style = adapter.adapt(heat_level="hot", pace="accelerating")
        assert style.max_tokens <= 256
        assert style.temperature == 0.7

    def test_cold_stable_allows_longer(self):
        adapter = StyleAdapter()
        style = adapter.adapt(heat_level="cold", pace="decelerating", topic_stability=0.8)
        assert style.max_tokens > 128
        assert style.max_tokens <= 1600

    def test_user_concise_override(self):
        adapter = StyleAdapter()
        style = adapter.adapt(heat_level="warm", pace="steady", user_communication_style="concise")
        assert style.max_tokens <= 80
        assert "1-2句话" in style.length_instruction
        assert style.temperature == 0.5

    def test_user_casual_increases_temperature(self):
        adapter = StyleAdapter()
        style = adapter.adapt(heat_level="warm", pace="steady", user_communication_style="casual")
        assert style.temperature == 0.8
        assert "轻松随意" in style.tone_instruction

    def test_overheated_very_short(self):
        adapter = StyleAdapter()
        style = adapter.adapt(heat_level="overheated", pace="accelerating")
        assert style.max_tokens <= 128


class TestResponseAssembler:
    def test_assemble_includes_all_sections(self):
        assembler = ResponseAssembler()
        msg = Message(role="human", content="我今天心情不好", speaker="u1")
        emotion = EmotionState(valence=-0.6, arousal=0.5, intensity=0.7)
        intent = IntentAnalysisV3(social_intent=SocialIntent.EMOTIONAL)

        bundle = assembler.assemble(
            message=msg,
            intent=intent,
            emotion=emotion,
            memories=[{"source": "working_memory", "content": "用户上周说工作压力大"}],
            group_profile=None,
            user_profile=None,
        )

        assert "你在一个多人聊天场景里" in bundle.system_prompt
        assert "当下的感觉" in bundle.system_prompt
        assert "相关记忆" in bundle.system_prompt
        assert "工作压力大" in bundle.system_prompt
        assert "我今天心情不好" in bundle.user_content

    def test_assemble_with_group_profile(self):
        assembler = ResponseAssembler()
        msg = Message(role="human", content="hello", speaker="u1")
        emotion = EmotionState(valence=0.2, arousal=0.3, intensity=0.5)
        intent = IntentAnalysisV3(social_intent=SocialIntent.SOCIAL)

        group = GroupSemanticProfile(group_id="g1", typical_interaction_style="humorous")
        group.atmosphere_history.append(AtmosphereSnapshot(
            timestamp="2026-04-17T10:00:00", group_valence=0.3, group_arousal=0.4
        ))

        bundle = assembler.assemble(
            message=msg,
            intent=intent,
            emotion=emotion,
            memories=[],
            group_profile=group,
            user_profile=None,
        )

        assert "群体风格" in bundle.system_prompt
        assert "轻松幽默" in bundle.system_prompt
        assert "群里氛围" in bundle.system_prompt

    def test_assemble_with_user_profile_concise(self):
        assembler = ResponseAssembler()
        msg = Message(role="human", content="test", speaker="u1")
        emotion = EmotionState(valence=0.0, arousal=0.0, intensity=0.0)
        intent = IntentAnalysisV3(social_intent=SocialIntent.SILENT)

        user = UserSemanticProfile(user_id="u1", communication_style="concise")
        group = GroupSemanticProfile(group_id="g1")

        bundle = assembler.assemble(
            message=msg,
            intent=intent,
            emotion=emotion,
            memories=[],
            group_profile=group,
            user_profile=user,
            heat_level="warm",
            pace="steady",
        )

        assert "1-2句话" in bundle.system_prompt

    def test_assemble_delayed(self):
        assembler = ResponseAssembler()
        bundle = assembler.assemble_delayed(
            message_content="刚才的话题很有趣",
            group_profile=GroupSemanticProfile(group_id="g1", typical_interaction_style="humorous"),
        )
        assert "话题有了自然间隙" in bundle.system_prompt
        assert "刚才的话题很有趣" in bundle.user_content
        assert "轻松幽默" in bundle.system_prompt

    def test_assemble_proactive(self):
        assembler = ResponseAssembler()
        bundle = assembler.assemble_proactive(
            trigger_reason="silence_30min",
            group_profile=GroupSemanticProfile(group_id="g1", interest_topics=["photography", "travel"]),
            suggested_tone="casual",
        )
        assert "silence_30min" in bundle.system_prompt
        assert "casual" in bundle.system_prompt
        assert "photography" in bundle.system_prompt
