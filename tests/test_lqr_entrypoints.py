"""Shell routing checks with a fake interpreter; no physics or training runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class LqrEntrypoints(unittest.TestCase):
    def route(self, script, args, existing=True):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);trace=base/'trace.jsonl';fake=base/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                            'with open(os.environ["TEST_TRACE"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            fake.chmod(0o755)
            artifact=base/'controller.npz'
            if existing: artifact.write_bytes(b'routing test only')
            env={**os.environ,'MJLAB_PYTHON':str(fake),'TEST_TRACE':str(trace),'WANGYU_LQR_FILE':str(artifact)}
            subprocess.run(['bash',str(ROOT/script),*args],env=env,check=True,capture_output=True,text=True)
            return [json.loads(line) for line in trace.read_text().splitlines()]

    def test_lqr_training_is_opt_in_and_help_does_not_prepare(self):
        default=self.route('train.sh',['--agent.max-iterations','2'])
        self.assertEqual(default[0][2],'Wangyu-Moon-Stand-Lunar-v15')
        self.assertEqual(self.route('train.sh',['stand-plane'])[0][2],'Wangyu-Moon-Stand-Flat-v14')
        self.assertEqual(self.route('run.sh',['stand-plane'])[0],['scripts/play_stand.py','--flat-v14'])
        self.assertEqual(self.route('train.sh',['stand-v13'])[0][2],'Wangyu-Moon-Stand-Flat-v13')
        self.assertEqual(self.route('run.sh',['stand-v13'])[0],['scripts/play_stand.py','--flat-v13'])
        self.assertEqual(self.route('train.sh',['stand-v12'])[0][2],'Wangyu-Moon-Stand-Support-Flat-v12')
        self.assertEqual(self.route('run.sh',['stand-v12'])[0],['scripts/play_stand.py','--flat-v12'])
        self.assertEqual(self.route('train.sh',['stand-v10'])[0][2],'Wangyu-Moon-Stand-Support-v10')
        self.assertEqual(self.route('run.sh',['stand-v10'])[0],['scripts/play_stand.py','--support-v10'])
        self.assertEqual(self.route('train.sh',['stand-v9'])[0][2],'Wangyu-Moon-Stand-Push-v9')
        self.assertEqual(self.route('train.sh',['stand-contact'])[0][2],'Wangyu-Moon-Stand-Support-Contact-v11')
        self.assertEqual(self.route('run.sh',['stand-contact'])[0],['scripts/play_stand.py','--contact-v11'])
        legacy=self.route('train.sh',['stand-v8'])
        self.assertEqual(legacy[0][2],'Wangyu-Moon-Stand-Push-v8')
        self.assertEqual(self.route('run.sh',['stand-v8'])[0],['scripts/play_stand.py','--push-v8'])
        old=self.route('train.sh',['stand-v4'])
        self.assertEqual(old[0][2],'Wangyu-Moon-Stand-Balance-v4')
        lqr=self.route('train.sh',['stand-lqr','--agent.max-iterations','2'])
        self.assertEqual(lqr[0][2],'Wangyu-Moon-Stand-LQR-Flat-v7')
        help_calls=self.route('train.sh',['stand-lqr','--help'],existing=False)
        self.assertEqual(len(help_calls),1)
        self.assertEqual(help_calls[0][0],'launch.py')

    def test_missing_artifact_prepares_once_and_play_selects_lqr(self):
        calls=self.route('train.sh',['stand-lqr'],existing=False)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0][0],'scripts/prepare_stand_lqr.py')
        self.assertEqual(calls[1][2],'Wangyu-Moon-Stand-LQR-Flat-v7')
        play=self.route('run.sh',['stand-lqr','--agent','zero'])
        self.assertEqual(play[0],['scripts/play_stand.py','--lqr-flat','--agent','zero'])
        terrain=self.route('train.sh',['stand-lqr-terrain'])
        self.assertEqual(terrain[0][2],'Wangyu-Moon-Stand-LQR-v5')


if __name__=='__main__': unittest.main()
