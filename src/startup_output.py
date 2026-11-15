"""Capture Python/native startup chatter without hiding runtime output or errors."""
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import sys
import tempfile


class StartupOutput:
    def __init__(self, verbose=False):
        self.enabled = not verbose
        self.path = None
        self._saved = None
        self._log = None

    @staticmethod
    def _flush():
        sys.stdout.flush()
        sys.stderr.flush()
        # MuJoCo/Warp can print through C stdio rather than Python streams.
        ctypes.CDLL(None).fflush(None)

    def start(self):
        if not self.enabled or self._saved is not None:
            return
        if self.path is None:
            fd, name = tempfile.mkstemp(prefix='wangyu-run-init-', suffix='.log')
            os.close(fd)
            self.path = Path(name)
        self._flush()
        self._log = self.path.open('ab', buffering=0)
        self._saved = (os.dup(1), os.dup(2))
        os.dup2(self._log.fileno(), 1)
        os.dup2(self._log.fileno(), 2)

    def resume(self):
        if self._saved is None:
            return
        self._flush()
        stdout, stderr = self._saved
        os.dup2(stdout, 1)
        os.dup2(stderr, 2)
        os.close(stdout)
        os.close(stderr)
        self._saved = None
        self._log.close()
        self._log = None

    @contextmanager
    def capture(self):
        self.start()
        try:
            yield
        finally:
            # Exceptions propagate after stderr is restored: full tracebacks remain visible.
            self.resume()


def run_play_quiet(output, task_id, cfg, *, env_cfg=None):
    """Restore terminal before the viewer loop; only initialization is captured."""
    from mjlab.scripts import play
    native, viser = play.NativeMujocoViewer, play.ViserPlayViewer
    load_env_cfg = play.load_env_cfg

    def configured_env(name, play=False):
        if name == task_id and env_cfg is not None:
            return env_cfg
        return load_env_cfg(name, play=play)

    class PlaybackNativeViewer(native):
        def setup(self):
            super().setup()
            import mujoco
            # Hide native red force arrows; keep actual applied wrenches intact.
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                self.mjm.vis.map.force = .01
                self.mjm.vis.scale.forcewidth = .04
            # Used by MJLab's additional-environment scenes as well.
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

        def _update_debug_visualizers(self, viewer):
            super()._update_debug_visualizers(viewer)
            from wrench_visualization import add_torque_geoms
            add_torque_geoms(viewer.user_scn,self.mjd)

    def native_ready(*args, **kwargs):
        if task_id in ('Wangyu-Moon-Stand-Lunar-v15','Wangyu-Moon-Hop-Lunar-v2'):
            from shared_moon_viewer import SharedMoonViewer
            viewer=SharedMoonViewer(*args,**kwargs)
            viewer._startup_output=output
            return viewer
        viewer = PlaybackNativeViewer(*args, **kwargs)
        output.resume()
        return viewer

    def viser_ready(*args, **kwargs):
        # Keep the server URL/login messages visible when choosing web visualization.
        output.resume()
        return viser(*args, **kwargs)

    with output.capture():
        play.load_env_cfg = configured_env
        play.NativeMujocoViewer = native_ready
        play.ViserPlayViewer = viser_ready
        try:
            return play.run_play(task_id, cfg)
        finally:
            play.NativeMujocoViewer, play.ViserPlayViewer = native, viser
            play.load_env_cfg = load_env_cfg
