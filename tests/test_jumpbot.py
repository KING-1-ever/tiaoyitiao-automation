import unittest
import threading
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from jumpbot import Engine, Uncertain, analyze, landing_error, Adb

FIXTURES = Path(__file__).parent/'fixtures'


class VisionTests(unittest.TestCase):
    def test_opening_geometry(self):
        scene = analyze(Image.open(FIXTURES/'opening.png'))
        self.assertAlmostEqual(scene.piece[0], 224, delta=3)
        self.assertAlmostEqual(scene.target[0], 526, delta=4)
        self.assertAlmostEqual(scene.target[1], 707, delta=5)

    def test_blank_screen_rejected(self):
        with self.assertRaises(Uncertain):
            analyze(Image.new('RGB', (720, 1558), '#c8cbd2'))

    def test_short_jump_clipped_platform(self):
        scene = analyze(Image.open(FIXTURES/'short-before.png'))
        self.assertAlmostEqual(scene.target[1], 751, delta=4)
        error, _, confidence = landing_error(scene, Image.open(FIXTURES/'short-after.png'))
        self.assertLess(abs(error), 25)
        self.assertGreater(confidence, .985)

    def test_short_round_platform_center_marker(self):
        scene = analyze(Image.open(FIXTURES/'short-round.png'))
        self.assertAlmostEqual(scene.target[0], 317, delta=3)
        self.assertAlmostEqual(scene.target[1], 756, delta=3)

    def test_pattern_changes_color_after_landing(self):
        scene = analyze(Image.open(FIXTURES/'pattern-before.png'))
        error, _, confidence = landing_error(scene, Image.open(FIXTURES/'pattern-after.png'))
        self.assertAlmostEqual(error, -11, delta=4)
        self.assertGreater(confidence, .95)

    def test_tiny_platform_landing(self):
        scene = analyze(Image.open(FIXTURES/'tiny-before.png'))
        error, _, confidence = landing_error(scene, Image.open(FIXTURES/'tiny-after.png'))
        self.assertAlmostEqual(error, -8.4, delta=3)
        self.assertGreater(confidence, .985)

    def test_landing_with_camera_translation(self):
        scene = analyze(Image.open(FIXTURES/'before.png'))
        error, movement, confidence = landing_error(scene, Image.open(FIXTURES/'after.png'))
        self.assertLess(abs(error), 25)
        self.assertGreater(abs(movement), 60)
        self.assertGreater(confidence, .985)


class ControlTests(unittest.TestCase):
    def engine(self):
        stop = threading.Event()
        engine = Engine(lambda *_: None, stop)
        engine.adb = Mock()
        return engine, stop

    def test_pause_before_loop_never_presses(self):
        engine, stop = self.engine()
        stop.set()
        engine.run(2, .5)
        engine.adb.press.assert_not_called()

    def test_pause_during_delay_never_presses(self):
        engine, stop = self.engine()
        engine.stable = Mock(return_value=Image.open(FIXTURES/'opening.png'))
        def wait(_):
            stop.set()
            raise InterruptedError
        engine.wait = wait
        with self.assertRaises(InterruptedError):
            engine.run(2, .5)
        engine.adb.press.assert_not_called()

    def test_changed_scene_never_presses(self):
        engine, _ = self.engine()
        engine.stable = Mock(return_value=Image.open(FIXTURES/'opening.png'))
        engine.adb.screenshot.return_value = Image.new('RGB', (720, 1558), 'white')
        engine.wait = lambda _: None
        with self.assertRaises(Uncertain):
            engine.run(2, .5)
        engine.adb.press.assert_not_called()

    def test_disconnection_never_presses(self):
        engine, _ = self.engine()
        engine.adb.connect.side_effect = Uncertain('disconnected')
        with self.assertRaises(Uncertain):
            engine.run(2, .5)
        engine.adb.press.assert_not_called()

    def test_other_app_rejected(self):
        adb = Adb()
        adb.run = Mock(return_value=b'mCurrentFocus=Window{12 u0 com.android.settings/.Settings}')
        with self.assertRaises(Uncertain):
            adb.check_app()

    def test_multiple_devices_rejected(self):
        adb = Adb()
        adb.run = Mock(return_value=b'List of devices attached\na device\nb device\n')
        with self.assertRaises(Uncertain):
            adb.connect()


if __name__ == '__main__':
    unittest.main()
