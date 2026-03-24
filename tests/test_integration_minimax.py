"""
Integration tests for MiniMax LLM provider.

These tests make real API calls to the MiniMax API.
Set MINIMAX_API_KEY and RUN_INTEGRATION_TESTS=1 to run.
"""

import os
import unittest


SKIP_REASON = "Set RUN_INTEGRATION_TESTS=1 and MINIMAX_API_KEY to run integration tests"


def _should_skip():
    return not (
        os.getenv("RUN_INTEGRATION_TESTS") == "1" and os.getenv("MINIMAX_API_KEY")
    )


@unittest.skipIf(_should_skip(), SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def setUp(self):
        os.environ["LLM_PROVIDER"] = "minimax"

    def tearDown(self):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("MINIMAX_MODEL", None)

    def test_minimax_chat_completion(self):
        """Test basic chat completion with MiniMax M2.5."""
        from llm_provider import get_chat_llm

        llm = get_chat_llm(temperature=0.01, max_tokens=100)
        response = llm.invoke("What is 2 + 2? Reply with just the number.")
        self.assertIn("4", response.content)

    def test_minimax_think_tags_stripped(self):
        """Test that <think> tags are stripped from responses."""
        from llm_provider import get_chat_llm

        llm = get_chat_llm(temperature=0.01, max_tokens=200)
        response = llm.invoke("What is the capital of France? Reply briefly.")
        # Think tags should be stripped
        self.assertNotIn("<think>", response.content)
        self.assertIn("Paris", response.content)

    def test_minimax_highspeed_model(self):
        """Test MiniMax M2.5-highspeed model."""
        from llm_provider import get_chat_llm

        os.environ["MINIMAX_MODEL"] = "MiniMax-M2.5-highspeed"
        llm = get_chat_llm(temperature=0.01, max_tokens=200)
        response = llm.invoke("Say hello in one sentence.")
        self.assertTrue(len(response.content) > 0)


if __name__ == "__main__":
    unittest.main()
