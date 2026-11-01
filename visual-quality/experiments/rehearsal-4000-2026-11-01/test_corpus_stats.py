"""Prevent source-metadata contamination and stale-statistics cache reuse."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import corpus_stats


class StatsCacheTests(unittest.TestCase):
    def test_identical_view_different_originals_have_separate_cache_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);view=root/'view.png';view.write_bytes(b'frozen-view')
            a=root/'a.jpg';a.write_bytes(b'original-a')
            b=root/'b.png';b.write_bytes(b'original-b')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            row={'image_path':str(view),'view_sha256':sha(view),'source_path':str(a),'source_sha256':sha(a)}
            def analyse(path,**kwargs):return {'sha256':sha(view),'source_path':kwargs['source_path']}
            with patch.object(corpus_stats,'DATA',root),patch.object(corpus_stats,'analyse_file',side_effect=analyse) as analyse_mock:
                first=corpus_stats.one(row)
                self.assertEqual(corpus_stats.one(row),first)
                second=corpus_stats.one({**row,'source_path':str(b),'source_sha256':sha(b)})
                self.assertEqual(analyse_mock.call_count,2)
                self.assertNotEqual(first['source_sha256'],second['source_sha256'])
                a.write_bytes(b'changed-source')
                with self.assertRaises(AssertionError):corpus_stats.one(row)
                self.assertEqual(analyse_mock.call_count,2)


if __name__=='__main__':unittest.main()
