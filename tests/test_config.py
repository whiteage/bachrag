import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cfg import PROJECT_ROOT, LLMConfig, Settings


class ConfigurationTests(unittest.TestCase):
    def test_deepseek_defaults_and_secret_hidden_from_repr(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-secret"}, clear=True):
            config = LLMConfig.from_env()
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.api_key, "test-secret")
        self.assertNotIn("test-secret", repr(config))

    def test_explicit_compatible_endpoint_and_key_override_provider(self):
        environment = {
            "LLM_PROVIDER": "openai", "OPENAI_API_KEY": "provider-test",
            "LLM_API_KEY": "override-test", "LLM_MODEL": "local-model",
            "LLM_BASE_URL": "http://localhost:8080/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = LLMConfig.from_env()
        self.assertEqual(config.api_key, "override-test")
        self.assertEqual(config.model, "local-model")
        self.assertEqual(config.base_url, "http://localhost:8080/v1")

    def test_default_paths_are_independent_of_working_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Settings()
        self.assertEqual(config.data_dir, PROJECT_ROOT / "runtime")
        self.assertEqual(config.upload_dir, config.data_dir / "uploads")

    def test_custom_data_directory_applies_to_all_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"BACHRAG_DATA_DIR": directory}, clear=True):
                config = Settings()
            for path in (config.upload_dir, config.index_path, config.data_path, config.state_file):
                self.assertEqual(path.parent, Path(directory).resolve())


if __name__ == "__main__":
    unittest.main()
