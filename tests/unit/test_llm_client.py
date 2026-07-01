"""Unit tests for the LLM client abstraction (mocked — no real API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestComplete:
    def test_uses_anthropic_when_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from src.settings.config import get_settings
        get_settings.cache_clear()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "ok"}')]

        with patch("anthropic.Anthropic") as MockAnthropic:
            MockAnthropic.return_value.messages.create.return_value = mock_response
            from src.llm import client as llm_client
            # Force re-import to pick up new settings
            import importlib
            importlib.reload(llm_client)
            result = llm_client.complete(system="sys", user="user")

        assert result == '{"result": "ok"}'
        get_settings.cache_clear()

    def test_raises_when_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        from src.settings.config import get_settings
        get_settings.cache_clear()

        import importlib
        from src.llm import client as llm_client
        importlib.reload(llm_client)

        with pytest.raises(RuntimeError, match="No LLM credentials"):
            llm_client.complete(system="sys", user="user")

        get_settings.cache_clear()

    def test_falls_back_to_azure_when_only_azure_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        from src.settings.config import get_settings
        get_settings.cache_clear()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"ok": true}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("openai.AzureOpenAI") as MockAzure:
            MockAzure.return_value.chat.completions.create.return_value = mock_response
            import importlib
            from src.llm import client as llm_client
            importlib.reload(llm_client)
            result = llm_client.complete(system="sys", user="user")

        assert result == '{"ok": true}'
        get_settings.cache_clear()
