"""Checkpoint discovery on fake files; no policy loading or simulation."""
from pathlib import Path
import tempfile
import unittest
from checkpoint_selection import select_experiment,latest_saved


class CheckpointSelection(unittest.TestCase):
    def test_disturbance_match_latest_and_task_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs=Path(tmp);base='task_v10'
            def save(exp,run,number):
                p=logs/exp/run/f'model_{number}.pt';p.parent.mkdir(parents=True,exist_ok=True);p.touch();return p
            save(base,'2026-01-01',100)
            expected=save(base+'_force200_torque50','2026-01-02',200)
            save(base+'_force200_torque50','2026-01-02',25)
            save(base+'_force200_torque50','2099_smoke',999)
            save('task_v9','2099-01-01',999)
            name,_=select_experiment(logs,base,force=200,torque=50)
            self.assertEqual(name,base+'_force200_torque50')
            self.assertEqual(latest_saved(logs/name),expected)
            self.assertEqual(select_experiment(logs,base)[0],name)
            self.assertEqual(select_experiment(logs,base,explicit=base)[0],base)
            self.assertEqual(select_experiment(logs,'missing_v11')[0],'missing_v11')

    def test_corrected_lunar_physics_prefers_matching_fixed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs=Path(tmp);base='wangyu_moon_stand_lunar_v15'
            for name in (base+'_force200_torque50',base+'_groundsync1_force200_torque50'):
                run=logs/name/'2026-11-14';run.mkdir(parents=True)
                (run/'model_799.pt').touch()
            selected,_=select_experiment(logs,base,force=200,torque=50)
            self.assertEqual(selected,base+'_groundsync1_force200_torque50')
            self.assertEqual(select_experiment(logs,base)[0],selected)
            detailed=base+'_groundsync1_detail1_force200_torque50'
            run=logs/detailed/'2026-11-14';run.mkdir(parents=True)
            (run/'model_799.pt').touch()
            self.assertEqual(select_experiment(logs,base,force=200,torque=50)[0],detailed)
            self.assertEqual(select_experiment(logs,base)[0],detailed)
            newest=base+'_groundsync1_detail3_force200_torque50'
            run=logs/newest/'2026-11-15';run.mkdir(parents=True)
            (run/'model_10.pt').touch()
            self.assertEqual(select_experiment(logs,base,force=200,torque=50)[0],newest)
            self.assertEqual(select_experiment(logs,base)[0],newest)


if __name__=='__main__':unittest.main()
