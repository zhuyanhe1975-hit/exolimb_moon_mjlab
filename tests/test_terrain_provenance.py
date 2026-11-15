import tempfile
import argparse
from pathlib import Path
import unittest
from unittest.mock import patch
import terrain_provenance as provenance


class TerrainProvenance(unittest.TestCase):
    def test_playback_transfer_default_and_strict_override(self):
        parser=argparse.ArgumentParser();provenance.add_playback_options(parser)
        self.assertTrue(parser.parse_args([]).terrain_transfer)
        self.assertFalse(parser.parse_args(['--strict-terrain']).terrain_transfer)
    def test_snapshot_match_missing_changed_and_corrupt(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'model_1.pt';checkpoint.touch()
            with self.assertRaises(ValueError):provenance.verify(checkpoint)
            provenance.save(Path(folder)/'terrain')
            provenance.verify(checkpoint)
            with patch.object(provenance,'fingerprint',return_value='changed'):
                with self.assertRaises(ValueError):provenance.verify(checkpoint)
                provenance.verify(checkpoint,allow_transfer=True)
            with (Path(folder)/'terrain/bank.npz').open('ab') as archive:archive.write(b'corrupt')
            with self.assertRaises(ValueError):provenance.verify(checkpoint)


if __name__=='__main__':unittest.main()
