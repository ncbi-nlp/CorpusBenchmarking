import unittest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from corpus_benchmark.runner import _load_corpus
from corpus_benchmark.models.config import BenchmarkConfig, LoaderSpec
from corpus_benchmark.models.corpus import BenchmarkCorpus

class TestRunnerAcquisition(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.cache_file = self.test_dir / "cache.json.gz"
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("corpus_benchmark.runner.LOADERS")
    @patch("corpus_benchmark.models.corpus.BenchmarkCorpus.from_json")
    def test_load_corpus_cache_behavior(self, mock_from_json, mock_loaders):
        workspace = MagicMock()
        benchmark_name = "test_corpus"
        config = BenchmarkConfig(
            name=benchmark_name,
            loader=LoaderSpec(name="dummy"),
            cache_filename=str(self.cache_file)
        )
        
        # Mock loader
        mock_loaders.__contains__.return_value = True
        mock_loader = MagicMock()
        mock_loaders.__getitem__.return_value = mock_loader
        mock_corpus = MagicMock()
        mock_loader.return_value = mock_corpus
        
        # 1. was_ready=True, cache exists -> load from cache
        workspace.acquisition_manager.ensure_corpus_ready.return_value = True
        self.cache_file.touch()
        
        _load_corpus(workspace, benchmark_name, config)
        
        workspace.acquisition_manager.ensure_corpus_ready.assert_called_with(benchmark_name, config)
        mock_from_json.assert_called_once()
        mock_loader.assert_not_called()
        
        mock_from_json.reset_mock()
        mock_loader.reset_mock()
        
        # 2. was_ready=False, cache exists -> ignore cache, load from source
        workspace.acquisition_manager.ensure_corpus_ready.return_value = False
        
        _load_corpus(workspace, benchmark_name, config)
        
        mock_from_json.assert_not_called()
        mock_loader.assert_called_once()
        mock_corpus.to_json.assert_called() # Should re-save cache

if __name__ == "__main__":
    unittest.main()
