import unittest
from pathlib import Path
import tempfile
import shutil
from corpus_benchmark.acquisition import AcquisitionManager
from corpus_benchmark.models.config import BenchmarkConfig, AcquisitionSpec, LoaderSpec, WorkspaceConfig

class TestAcquisition(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.corpora_dir = self.test_dir / "corpora"
        self.corpora_dir.mkdir()
        self.workspace_config = WorkspaceConfig(
            corpora_download_dir=str(self.corpora_dir)
        )
        self.manager = AcquisitionManager(self.workspace_config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sentinel_file(self):
        # Create a dummy config that doesn't actually need to download anything
        # but has an acquisition spec.
        
        corpus_name = "test_corpus"
        config = BenchmarkConfig(
            name=corpus_name,
            loader=LoaderSpec(name="dummy", params={"path": str(self.corpora_dir / corpus_name / "data.txt")}),
            acquisition=AcquisitionSpec(source_urls=["http://example.com/data.txt"], format="none")
        )
        
        corpus_dir = self.corpora_dir / corpus_name
        corpus_dir.mkdir()
        data_file = corpus_dir / "data.txt"
        data_file.touch()
        
        # Initially, all_exist is True, but sentinel is missing.
        # It should NOT be ready if it has an acquisition spec.
        
        from unittest.mock import patch
        with patch("urllib.request.urlretrieve") as mock_retrieve:
            # 1. First call without sentinel
            was_ready = self.manager.ensure_corpus_ready(corpus_name, config)
            self.assertFalse(was_ready, "Should not be ready without sentinel")
            self.assertTrue((corpus_dir / ".acquisition_done").exists(), "Sentinel should be created")
            
            # 2. Second call with sentinel
            was_ready = self.manager.ensure_corpus_ready(corpus_name, config)
            self.assertTrue(was_ready, "Should be ready with sentinel")
            
            # 3. Remove sentinel, should not be ready again
            (corpus_dir / ".acquisition_done").unlink()
            was_ready = self.manager.ensure_corpus_ready(corpus_name, config)
            self.assertFalse(was_ready, "Should not be ready after removing sentinel")

    def test_no_acquisition_spec(self):
        # If no acquisition spec, we trust all_exist.
        corpus_name = "manual_corpus"
        config = BenchmarkConfig(
            name=corpus_name,
            loader=LoaderSpec(name="dummy", params={"path": str(self.corpora_dir / corpus_name / "data.txt")}),
            acquisition=None
        )
        
        corpus_dir = self.corpora_dir / corpus_name
        corpus_dir.mkdir()
        data_file = corpus_dir / "data.txt"
        data_file.touch()
        
        was_ready = self.manager.ensure_corpus_ready(corpus_name, config)
        self.assertTrue(was_ready, "Should be ready without acquisition spec if files exist")
        self.assertFalse((corpus_dir / ".acquisition_done").exists(), "Sentinel should NOT be created for manual corpus")

if __name__ == "__main__":
    unittest.main()
