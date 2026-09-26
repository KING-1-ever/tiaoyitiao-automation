"""有界真机验证。默认只识别；--jumps N 才执行至多 N 次跳跃。"""
import argparse
import json
import threading
from jumpbot import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jumps', type=int, default=0)
    args = parser.parse_args()
    root = ROOT/'diagnostics'
    root.mkdir(exist_ok=True)
    settings_path = ROOT/'settings.json'
    settings = json.loads(settings_path.read_text('utf-8')) if settings_path.exists() else {'coefficient': 2., 'delay': .5}
    def emit(kind, value):
        if kind == 'image':
            value.save(root/f'aim-{time.time_ns()}.png')
        else:
            print(kind, value, flush=True)
            with (root/'validation.log').open('a', encoding='utf-8') as f:
                f.write(f'{kind}: {value}\n')
            if kind == 'coefficient':
                settings[kind] = value
                settings_path.write_text(json.dumps(settings, indent=2), 'utf-8')
    engine = Engine(emit, threading.Event())
    try:
        if args.jumps > 0:
            engine.run(settings['coefficient'], settings['delay'], args.jumps)
        else:
            print(engine.adb.connect())
            shot = engine.stable()
            scene = analyze(shot)
            print(scene.piece, scene.target, scene.distance)
            annotate(scene).save(root/'preview.png')
    finally:
        if engine.adb.serial:
            engine.adb.screenshot().save(root/'latest.png')
