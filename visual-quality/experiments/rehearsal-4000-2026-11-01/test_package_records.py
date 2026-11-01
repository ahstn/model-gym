"""Synthetic final-record packaging checks; never read live trial journals."""
import gzip
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import package_records as records


class PackageRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'attempts/phase/case/01').mkdir(parents=True)
        self.receipt = self.source / 'attempts/phase/case/01/result.json'
        self.receipt.write_bytes(b'{"received_at":"2026-11-01T12:34:56+00:00","score":3.75}\n')
        (self.source / 'plan.json').write_text('{"planned":true}\n')
        (self.source / '.driver.lock').write_text('transient lock')
        (self.source / 'driver.pid').write_text('123')
        (self.source / 'tmp').mkdir()
        (self.source / 'tmp/partial.json').write_text('unfinished')
        self.output = self.root / 'execution-records'

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, output=None, secrets=None, stopped=True):
        return records.package({'model': self.source}, output or self.output,
                               all_runs_stopped=stopped, secrets=secrets or {})

    def test_roundtrip_deterministic_headers_transients_and_idempotency(self):
        index = self.package()
        archive = self.output / 'model.tar.gz'
        original = archive.read_bytes()
        self.assertEqual(original[4:8], b'\0\0\0\0')
        self.assertEqual(index['archives'][0]['excluded_transients'], ['.driver.lock', 'driver.pid', 'tmp/partial.json'])
        with tarfile.open(archive, 'r:gz') as stream:
            self.assertEqual(stream.getnames(), ['attempts/phase/case/01/result.json', 'plan.json'])
            self.assertEqual(stream.extractfile(stream.getmembers()[0]).read(), self.receipt.read_bytes())
            for member in stream.getmembers():
                self.assertTrue(member.isfile())
                self.assertEqual((member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname), (0o644, 0, 0, 0, '', ''))
        self.assertEqual(self.package(), index)
        self.assertEqual(archive.read_bytes(), original)
        self.package(self.root / 'independent-output')
        self.assertEqual((self.root / 'independent-output/model.tar.gz').read_bytes(), original)
        self.assertEqual(json.loads((self.output / 'index.json').read_text()), index)

    def test_stopped_assertion_and_changed_sources_are_required(self):
        with self.assertRaisesRegex(ValueError, 'stopped'):
            self.package(stopped=False)
        snapshot = records.snapshot
        calls = 0
        def changed(root, secrets):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.receipt.write_text('{"changed_after_archiving":true}\n')
            return snapshot(root, secrets)
        with patch.object(records, 'snapshot', side_effect=changed), self.assertRaisesRegex(ValueError, 'snapshot changed'):
            self.package()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.execution-records-*')), [])

    def test_symlinks_and_destination_inside_source_are_rejected(self):
        (self.source / 'outside-link').symlink_to(self.root / 'outside')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.package()
        (self.source / 'outside-link').unlink()
        with self.assertRaisesRegex(ValueError, 'outside every source root'):
            self.package(self.source / 'archive')
        link = self.root / 'source-link'
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            records.package({'model': link}, self.output, all_runs_stopped=True, secrets={})
        with self.assertRaisesRegex(ValueError, 'safe simple names'):
            records.package({'../escape': self.source}, self.output, all_runs_stopped=True, secrets={})

    def test_exact_credentials_and_obvious_patterns_reject_without_value(self):
        sensitive = b'unit-test-private-token-value'
        self.receipt.write_bytes(b'authorization: ' + sensitive)
        with self.assertRaises(ValueError) as raised:
            self.package(secrets={'RUN_INFRA_API_KEY': sensitive})
        self.assertIn('RUN_INFRA_API_KEY', str(raised.exception))
        self.assertNotIn(sensitive.decode(), str(raised.exception))
        self.assertFalse(self.output.exists())
        for secret in (b'sk-or-v1-abc123xyz4567890', b'hf_abcdefghijklmnop', b'rpa_abcdefghijklmnop'):
            self.receipt.write_bytes(secret)
            with self.subTest(pattern=secret.split(b'_')[0]), self.assertRaises(ValueError) as raised:
                self.package()
            self.assertNotIn(secret.decode(), str(raised.exception))

    def test_existing_archives_are_never_overwritten_on_mismatch(self):
        self.package()
        original = (self.output / 'model.tar.gz').read_bytes()
        self.receipt.write_text('{"later_source":true}\n')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            self.package()
        self.assertEqual((self.output / 'model.tar.gz').read_bytes(), original)
        self.receipt.write_bytes(b'{"received_at":"2026-11-01T12:34:56+00:00","score":3.75}\n')
        (self.output / 'model.tar.gz').write_bytes(gzip.compress(b'changed archive'))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.package()

    def test_encrypted_substrings_do_not_bypass_exact_secret_or_token_checks(self):
        token = b'hf_' + b'abcdefghijklmnop'
        opaque = b'opaque_' + token + b'_ciphertext'
        records.scan('response.json', opaque, {})
        with self.assertRaisesRegex(ValueError, 'HF_TOKEN'):
            records.scan('response.json', opaque, {'HF_TOKEN': token})
        with self.assertRaisesRegex(ValueError, 'pattern:hf'):
            records.scan('response.json', b'{"token":"' + token + b'"}', {})

    def test_size_limits_fail_without_final_artifacts(self):
        with patch.object(records, 'MAX_FILE_BYTES', 8), self.assertRaisesRegex(ValueError, 'exceeds 100 MB'):
            self.package()
        with patch.object(records, 'MAX_ARCHIVE_BYTES', 64), self.assertRaisesRegex(ValueError, 'split by phase'):
            self.package()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
