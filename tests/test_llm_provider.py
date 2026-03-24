"""Unit tests for the multi-provider LLM factory."""

import os
import unittest
from unittest.mock import patch, MagicMock

from llm_provider import (
    get_chat_llm,
    get_embeddings,
    _get_provider,
    _clamp_temperature,
    _strip_think_tags,
    ChatMiniMax,
    MINIMAX_BASE_URL,
    MINIMAX_DEFAULT_MODEL,
    MINIMAX_MODELS,
)


class TestGetProvider(unittest.TestCase):
    """Tests for _get_provider helper."""

    def test_default_provider_is_openai(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROVIDER", None)
            self.assertEqual(_get_provider(), "openai")

    def test_explicit_openai_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            self.assertEqual(_get_provider(), "openai")

    def test_minimax_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}):
            self.assertEqual(_get_provider(), "minimax")

    def test_provider_case_insensitive(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "MiniMax"}):
            self.assertEqual(_get_provider(), "minimax")

    def test_provider_strips_whitespace(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "  minimax  "}):
            self.assertEqual(_get_provider(), "minimax")


class TestClampTemperature(unittest.TestCase):
    """Tests for MiniMax temperature clamping."""

    def test_zero_clamped_to_epsilon(self):
        result = _clamp_temperature(0)
        self.assertAlmostEqual(result, 0.01)

    def test_negative_clamped_to_epsilon(self):
        result = _clamp_temperature(-0.5)
        self.assertAlmostEqual(result, 0.01)

    def test_valid_temperature_unchanged(self):
        self.assertAlmostEqual(_clamp_temperature(0.5), 0.5)

    def test_one_unchanged(self):
        self.assertAlmostEqual(_clamp_temperature(1.0), 1.0)

    def test_above_one_clamped(self):
        self.assertAlmostEqual(_clamp_temperature(1.5), 1.0)

    def test_small_positive_unchanged(self):
        self.assertAlmostEqual(_clamp_temperature(0.01), 0.01)


class TestStripThinkTags(unittest.TestCase):
    """Tests for think-tag stripping."""

    def test_strips_simple_think_block(self):
        text = "<think>reasoning here</think>\nThe answer is 42."
        self.assertEqual(_strip_think_tags(text), "The answer is 42.")

    def test_strips_multiline_think_block(self):
        text = "<think>\nStep 1: ...\nStep 2: ...\n</think>\n\nParis"
        self.assertEqual(_strip_think_tags(text), "Paris")

    def test_no_think_tags_unchanged(self):
        text = "The capital of France is Paris."
        self.assertEqual(_strip_think_tags(text), text)

    def test_empty_think_block(self):
        text = "<think></think>Hello"
        self.assertEqual(_strip_think_tags(text), "Hello")

    def test_strips_json_after_think(self):
        text = '<think>Let me think...</think>\n{"answer": "Paris"}'
        self.assertEqual(_strip_think_tags(text), '{"answer": "Paris"}')

    def test_preserves_content_without_tags(self):
        text = '{"key": "value"}'
        self.assertEqual(_strip_think_tags(text), '{"key": "value"}')


class TestGetChatLLMOpenAI(unittest.TestCase):
    """Tests for get_chat_llm with OpenAI provider."""

    @patch("llm_provider.ChatOpenAI")
    def test_openai_default_params(self, mock_chat):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROVIDER", None)
            get_chat_llm()
            mock_chat.assert_called_once_with(
                temperature=0, model_name="gpt-4o", max_tokens=2000
            )

    @patch("llm_provider.ChatOpenAI")
    def test_openai_custom_params(self, mock_chat):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            get_chat_llm(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=1000)
            mock_chat.assert_called_once_with(
                temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=1000
            )


class TestGetChatLLMMiniMax(unittest.TestCase):
    """Tests for get_chat_llm with MiniMax provider."""

    @patch("llm_provider.ChatMiniMax")
    def test_minimax_default_model(self, mock_chat):
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            get_chat_llm()
            mock_chat.assert_called_once_with(
                temperature=0.01,
                model=MINIMAX_DEFAULT_MODEL,
                max_tokens=2000,
                openai_api_key="test-key",
                openai_api_base=MINIMAX_BASE_URL,
            )

    @patch("llm_provider.ChatMiniMax")
    def test_minimax_custom_model(self, mock_chat):
        env = {
            "LLM_PROVIDER": "minimax",
            "MINIMAX_API_KEY": "test-key",
            "MINIMAX_MODEL": "MiniMax-M2.5-highspeed",
        }
        with patch.dict(os.environ, env, clear=True):
            get_chat_llm(temperature=0.5, max_tokens=4000)
            mock_chat.assert_called_once_with(
                temperature=0.5,
                model="MiniMax-M2.5-highspeed",
                max_tokens=4000,
                openai_api_key="test-key",
                openai_api_base=MINIMAX_BASE_URL,
            )

    def test_minimax_missing_api_key_raises(self):
        env = {"LLM_PROVIDER": "minimax"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with self.assertRaises(ValueError) as ctx:
                get_chat_llm()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("llm_provider.ChatMiniMax")
    def test_minimax_temperature_clamping_zero(self, mock_chat):
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            get_chat_llm(temperature=0)
            call_kwargs = mock_chat.call_args[1]
            self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("llm_provider.ChatMiniMax")
    def test_minimax_base_url(self, mock_chat):
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            get_chat_llm()
            call_kwargs = mock_chat.call_args[1]
            self.assertEqual(call_kwargs["openai_api_base"], "https://api.minimax.io/v1")

    @patch("llm_provider.ChatMiniMax")
    def test_minimax_returns_chatminimax_instance(self, mock_chat):
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            result = get_chat_llm()
            # Verify ChatMiniMax was called (not plain ChatOpenAI)
            mock_chat.assert_called_once()


class TestChatMiniMaxSubclass(unittest.TestCase):
    """Tests for ChatMiniMax being a proper subclass."""

    def test_is_subclass_of_chatopenai(self):
        from langchain_openai import ChatOpenAI
        self.assertTrue(issubclass(ChatMiniMax, ChatOpenAI))


class TestGetEmbeddings(unittest.TestCase):
    """Tests for get_embeddings."""

    @patch("llm_provider.OpenAIEmbeddings")
    def test_returns_openai_embeddings(self, mock_emb):
        result = get_embeddings()
        mock_emb.assert_called_once()


class TestConstants(unittest.TestCase):
    """Tests for module constants."""

    def test_minimax_base_url(self):
        self.assertEqual(MINIMAX_BASE_URL, "https://api.minimax.io/v1")

    def test_minimax_default_model(self):
        self.assertEqual(MINIMAX_DEFAULT_MODEL, "MiniMax-M2.5")

    def test_minimax_models_list(self):
        self.assertIn("MiniMax-M2.5", MINIMAX_MODELS)
        self.assertIn("MiniMax-M2.5-highspeed", MINIMAX_MODELS)
        self.assertEqual(len(MINIMAX_MODELS), 2)


class TestProviderSwitching(unittest.TestCase):
    """Tests for runtime provider switching."""

    @patch("llm_provider.ChatMiniMax")
    @patch("llm_provider.ChatOpenAI")
    def test_switch_from_openai_to_minimax(self, mock_openai, mock_minimax):
        # First call with OpenAI
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            get_chat_llm()
            mock_openai.assert_called_once()

        # Second call with MiniMax
        env = {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            get_chat_llm()
            mock_minimax.assert_called_once()
            call_kwargs = mock_minimax.call_args[1]
            self.assertIn("openai_api_base", call_kwargs)


if __name__ == "__main__":
    unittest.main()
