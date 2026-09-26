"""QQ 跳一跳：仅通过 ADB 截图和触屏输入运行。坐标统一缩放至 720 宽。"""
from __future__ import annotations

import io
import ctypes
import json
import math
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent
WIDTH = 720
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def configure_dpi():
    if hasattr(ctypes, 'windll'):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


class Uncertain(RuntimeError):
    pass


@dataclass
class Scene:
    piece: tuple[float, float]
    target: tuple[float, float]
    radius: float
    image: Image.Image

    @property
    def distance(self):
        return math.dist(self.piece, self.target)


def normalized(image):
    return image.convert('RGB').resize((WIDTH, round(image.height * WIDTH / image.width)))


def find_piece(image):
    a = np.asarray(image).astype(np.int16)
    h, w = a.shape[:2]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mask = ((r > 25) & (r < 105) & (g > 20) & (g < 100)
            & (b > 45) & (b < 155) & (b-g > 12) & (b-r > 3)).astype('uint8')
    mask[:int(h*.36)] = 0
    mask[int(h*.78):] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 20 <= bw <= 75 and 55 <= bh <= 165 and area > 1400 and bh > bw:
            yy, xx = np.where(labels == i)
            bottom = yy.max()
            bx = float(np.median(xx[yy >= bottom-7]))
            candidates.append((area, (bx, float(bottom-3))))
    if len(candidates) != 1:
        raise Uncertain('棋子识别不唯一或未找到，请确认已进入可起跳画面。')
    return candidates[0][1]


def analyze(original):
    image = normalized(original)
    a = np.asarray(image).astype(np.int16)
    h, w = a.shape[:2]
    px, py = find_piece(image)
    # 每行从目标对侧边缘取背景，避免大平台伸出屏幕时污染背景估计。
    bg = np.median(a[:, :18] if px < w/2 else a[:, -18:], axis=1)
    diff = np.linalg.norm(a-bg[:, None, :], axis=2)
    mask = (diff > 24).astype('uint8')
    lo = max(int(h*.32), int(py-400))
    hi = min(int(py-30), int(h*.73))
    mask[:lo] = 0
    mask[hi:] = 0
    if px < w/2:
        mask[:, :int(px+65)] = 0
    else:
        mask[:, int(px-65):] = 0
    mask[:, :20] = 0
    mask[:, -20:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw >= 45 and bh >= 28 and area >= 1100 and y > lo+2:
            candidates.append((y, i))
    if not candidates:
        raise Uncertain('未可靠识别下一平台，已暂停。')
    candidates.sort()
    top, idx = candidates[0]
    yy, xx = np.where(labels == idx)
    top_x = float(np.median(xx[yy <= top+5]))
    rows = []
    for y in range(top+3, min(hi, top+170)):
        xs = np.flatnonzero(labels[y] == idx)
        if len(xs) >= 5:
            rows.append((y, int(xs[-1])))
    if len(rows) < 20:
        raise Uncertain('平台边界不足，已暂停。')
    # 投影中平台右边界在顶面中线处达到最右侧，左侧阴影不参与定位。
    right = max(x for _, x in rows)
    center_y = float(min(y for y, x in rows if x >= right-1))
    half_width = right-top_x
    roi_right = w-20 if px < w/2 else min(w-20, int(px-65))
    if right >= roi_right-2:
        center_y = py-abs(top_x-px)/math.sqrt(3)
        half_width = (center_y-top)*math.sqrt(3)
    else:
        if abs((center_y-top)-half_width/math.sqrt(3)) > max(22, half_width*.24):
            raise Uncertain('平台顶面形状异常，已暂停。')
        center_y = top+half_width/math.sqrt(3)
    if not 22 <= half_width <= 210 or not 12 <= center_y-top <= 125:
        raise Uncertain('平台几何形状不可信，已暂停。')
    # 连续中心跳跃时游戏会显示白色小圆点，可靠检测后优先使用其真实中心。
    white = ((a.min(axis=2) > 230) & (a.max(axis=2)-a.min(axis=2) < 18)).astype('uint8')
    wx0, wx1 = max(0, int(top_x-35)), min(w, int(top_x+36))
    wy0, wy1 = max(top+8, int(center_y-30)), min(hi, int(center_y+31))
    crop = white[wy0:wy1, wx0:wx1]
    if crop.size:
        nwhite, _, white_stats, centers = cv2.connectedComponentsWithStats(crop)
        dots = []
        for i in range(1, nwhite):
            x, y, bw, bh, area = white_stats[i]
            if 12 <= bw <= 38 and 6 <= bh <= 24 and 50 <= area <= 650 and 1.3 < bw/bh < 2.6:
                if x > 0 and y > 0 and x+bw < crop.shape[1] and y+bh < crop.shape[0]:
                    dots.append((float(centers[i][0]+wx0), float(centers[i][1]+wy0)))
        if len(dots) == 1 and abs(dots[0][0]-top_x) < 18:
            top_x, center_y = dots[0]
    target = (top_x, center_y)
    distance = math.dist((px, py), target)
    if not 65 <= distance <= 500:
        raise Uncertain('跳跃距离超出可靠范围，已暂停。')
    if abs(abs(target[0]-px)/math.sqrt(3)-(py-center_y)) > 48:
        raise Uncertain('棋子与平台的相对位置异常，已暂停。')
    return Scene((px, py), target, half_width, image)


def annotate(scene):
    im = scene.image.copy()
    draw = ImageDraw.Draw(im)
    draw.line((*scene.piece, *scene.target), fill='#ee5555', width=3)
    for (x, y), color in [(scene.piece, '#22dd77'), (scene.target, '#ee5555')]:
        draw.ellipse((x-7, y-7, x+7, y+7), outline=color, width=3)
        draw.line((x-13, y, x+13, y), fill=color, width=2)
        draw.line((x, y-13, x, y+13), fill=color, width=2)
    return im


def landing_error(scene, landed):
    """联合跟踪平台两侧，补偿相机平移，并避开落地后棋子遮挡的中心。"""
    after = normalized(landed)
    tx, ty = scene.target
    radius = scene.radius
    x0, x1 = max(0, round(tx-radius-10)), min(WIDTH, round(tx+radius+10))
    y0, y1 = round(ty-15), round(ty+min(100, radius*.8))
    before = np.array(scene.image)
    new = np.array(after)
    template = before[y0:y1, x0:x1]
    lp = find_piece(after)
    expected_dx, expected_dy = round(lp[0]-tx), round(lp[1]-ty)
    sx0, sx1 = max(0, x0+expected_dx-60), min(WIDTH, x1+expected_dx+60)
    sy0, sy1 = max(0, y0+expected_dy-45), min(len(new), y1+expected_dy+45)
    if sx1-sx0 < x1-x0 or sy1-sy0 < y1-y0:
        raise Uncertain('落地平台边界超出屏幕，已暂停。')
    bg = np.median(np.concatenate((before[y0:y1, :18], before[y0:y1, -18:]), axis=1), axis=1)
    mask = (np.linalg.norm(template.astype(float)-bg[:, None, :], axis=2) > 40).astype('uint8')*255
    mask[:, np.abs(np.arange(x0, x1)-tx) < max(40, radius*.60)] = 0
    old_px, old_py = scene.piece
    covered = ((np.arange(y0, y1)[:, None] >= old_py-155)
               & (np.arange(y0, y1)[:, None] <= old_py+8)
               & (np.abs(np.arange(x0, x1)[None, :]-old_px) < 40))
    mask[covered] = 0
    if np.count_nonzero(mask) < 150:
        raise Uncertain('平台跟踪纹理不足，已暂停。')
    scores = cv2.matchTemplate(new[sy0:sy1, sx0:sx1], template, cv2.TM_SQDIFF_NORMED, mask=mask)
    minimum, _, location, _ = cv2.minMaxLoc(scores)
    confidence = 1-minimum
    verified_edges = False
    if math.isfinite(confidence) and .95 <= confidence < .985:
        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        edges_before = cv2.Canny(gray_before, 15, 40)[y0:y1, x0:x1]
        edges_after = cv2.Canny(gray_after, 15, 40)[sy0:sy1, sx0:sx1]
        edges_before = cv2.GaussianBlur(edges_before, (5, 5), 0)
        edges_after = cv2.GaussianBlur(edges_after, (5, 5), 0)
        if np.count_nonzero(edges_before[mask > 0]) > 80:
            edge_scores = cv2.matchTemplate(edges_after, edges_before, cv2.TM_CCORR_NORMED, mask=mask)
            edge_scores = np.nan_to_num(edge_scores, nan=0., posinf=0., neginf=0.)
            _, edge_confidence, _, edge_location = cv2.minMaxLoc(edge_scores)
            verified_edges = edge_confidence > .80 and math.dist(edge_location, location) <= 3
            if edge_confidence > .90 and abs(edge_location[0]-location[0]) <= 3 and math.dist(edge_location, location) <= 30:
                verified_edges = True
                location = edge_location
    if not math.isfinite(confidence) or (confidence < .985 and not verified_edges):
        raise Uncertain(f'落地平台跟踪不可靠（{confidence:.2f}），已暂停。')
    dx, dy = location[0]+sx0-x0, location[1]+sy0-y0
    shifted_target = (tx+dx, ty+dy)
    error_x, error_y = lp[0]-shifted_target[0], lp[1]-shifted_target[1]
    if abs(error_y) > max(20, radius*.32):
        raise Uncertain('未确认棋子站稳在目标平台上，已暂停。')
    return error_x, lp[0]-(scene.piece[0]+dx), confidence


class Adb:
    def __init__(self):
        self.exe = ROOT/'tools'/'platform-tools'/'adb.exe'
        self.serial = None

    def run(self, *args, timeout=15):
        cmd = [str(self.exe)]
        if self.serial:
            cmd += ['-s', self.serial]
        p = subprocess.run(cmd+list(args), capture_output=True, timeout=timeout,
                           creationflags=NO_WINDOW)
        if p.returncode:
            raise Uncertain(p.stderr.decode('utf-8', 'replace').strip() or 'ADB 命令失败')
        return p.stdout

    def connect(self):
        self.serial = None
        lines = self.run('devices').decode().splitlines()[1:]
        devices = [line.split() for line in lines if line.strip()]
        if len(devices) != 1 or devices[0][1] != 'device':
            raise Uncertain('请只连接一台手机，开启 USB 调试并允许电脑授权。')
        self.serial = devices[0][0]
        model = self.run('shell', 'getprop', 'ro.product.model').decode().strip()
        return model

    def check_app(self):
        out = self.run('shell', 'dumpsys', 'window').decode('utf-8', 'replace')
        focus = [line for line in out.splitlines() if 'mCurrentFocus=' in line]
        if not focus or 'com.tencent.mobileqq/' not in focus[0]:
            raise Uncertain('QQ 不在前台或屏幕已锁定，请返回游戏后重试。')

    def screenshot(self):
        return Image.open(io.BytesIO(self.run('exec-out', 'screencap', '-p'))).convert('RGB')

    def press(self, size, milliseconds):
        x, y = round(size[0]*.5), round(size[1]*.78)
        self.run('shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(milliseconds),
                 timeout=max(10, milliseconds/1000+5))


class Engine:
    def __init__(self, emit, stop):
        self.adb = Adb()
        self.emit = emit
        self.stop = stop

    def wait(self, seconds):
        if self.stop.wait(seconds):
            raise InterruptedError

    def stable(self):
        previous = None
        for _ in range(12):
            if self.stop.is_set():
                raise InterruptedError
            self.adb.check_app()
            shot = self.adb.screenshot()
            small = np.asarray(normalized(shot).resize((180, round(shot.height/shot.width*180))))
            crop = small[int(len(small)*.38):int(len(small)*.78)].astype(float)
            if previous is not None and np.mean(np.abs(crop-previous)) < 1.3:
                return shot
            previous = crop
            self.wait(.25)
        raise Uncertain('画面持续变化，等待落稳超时。')

    def run(self, coefficient, delay, limit=0):
        model = self.adb.connect()
        self.emit('status', f'已连接 {model}')
        count = 0
        while not self.stop.is_set():
            shot = self.stable()
            scene = analyze(shot)
            self.emit('image', annotate(scene))
            duration = round(scene.distance*coefficient)
            if not 100 <= duration <= 1100:
                raise Uncertain('按压时长超出安全范围。')
            self.emit('status', f'第 {count+1} 跳：距离 {scene.distance:.1f}，按压 {duration} ms')
            self.wait(delay)
            self.adb.check_app()
            # 等待期间有手动操作或切换游戏页面时，不使用过期坐标。
            fresh = self.adb.screenshot()
            current = analyze(fresh)
            if math.dist(scene.piece, current.piece) > 5 or math.dist(scene.target, current.target) > 5:
                raise Uncertain('等待期间画面改变，请重新开始。')
            if self.stop.is_set():
                break
            self.adb.press(fresh.size, duration)
            count += 1
            self.emit('count', count)
            self.wait(.85)
            landed = self.stable()
            diagnostic = ROOT/'diagnostics'
            diagnostic.mkdir(exist_ok=True)
            fresh.save(diagnostic/'before.png')
            landed.save(diagnostic/'after.png')
            error_x, displacement_x, confidence = landing_error(scene, landed)
            direction = 1 if scene.target[0] > scene.piece[0] else -1
            actual_dx = displacement_x*direction
            wanted_dx = abs(scene.target[0]-scene.piece[0])
            error = error_x*direction
            if abs(error) > min(45, scene.radius*.55):
                raise Uncertain(f'落点偏离中心 {error:+.1f} 像素，停止以便重新校准。')
            if actual_dx > 50:
                proposal = coefficient*wanted_dx/actual_dx
                if .80 < proposal < 2.5:
                    coefficient += float(np.clip((proposal-coefficient)*.35, -.06, .06))
                    self.emit('coefficient', coefficient)
            self.emit('status', f'第 {count} 跳落点横向偏差 {error:+.1f} 像素')
            if limit and count >= limit:
                self.emit('status', f'{count} 次试跳完成，参数已保存；可点击开始连续运行。')
                break


class App:
    def __init__(self, root):
        self.root = root
        root.title('QQ 跳一跳助手')
        root.geometry('640x900')
        root.minsize(600, 780)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.closing = False
        self.photo = None
        self.settings = {'coefficient': 2.0, 'delay': .5}
        try:
            saved = json.loads((ROOT/'settings.json').read_text('utf-8'))
            coefficient, delay = float(saved['coefficient']), float(saved['delay'])
            if .8 <= coefficient <= 2.5 and .2 <= delay <= 10:
                self.settings.update(coefficient=coefficient, delay=delay)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.coeff = tk.StringVar(value=f"{self.settings['coefficient']:.4f}")
        self.delay = tk.StringVar(value=str(self.settings['delay']))
        self.status = tk.StringVar(value='请打开 QQ 跳一跳，先点「检查画面」。')
        self.count = tk.StringVar(value='本次已发起 0 跳')
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='QQ 跳一跳助手', font=('Microsoft YaHei UI', 17, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='绿色：棋子脚底  ·  红色：目标中心').pack(anchor='w', pady=5)
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=8)
        self.buttons = []
        for label, mode in [('检查画面', 'preview'), ('校准试跳（3次）', 'calibrate'), ('开始', 'run')]:
            b = ttk.Button(controls, text=label, command=lambda m=mode: self.start(m))
            b.pack(side='left', padx=3)
            self.buttons.append(b)
        ttk.Button(controls, text='暂停', command=self.pause).pack(side='left', padx=3)
        opts = ttk.Frame(frame)
        opts.pack(fill='x')
        ttk.Label(opts, text='落稳后等待（秒）').pack(side='left')
        ttk.Entry(opts, textvariable=self.delay, width=6).pack(side='left', padx=4)
        ttk.Label(opts, text='按压系数').pack(side='left')
        ttk.Entry(opts, textvariable=self.coeff, width=8).pack(side='left', padx=4)
        ttk.Label(frame, textvariable=self.count).pack(anchor='w', pady=(8, 2))
        ttk.Label(frame, textvariable=self.status, wraplength=520).pack(anchor='w')
        self.preview = ttk.Label(frame, anchor='center')
        self.preview.pack(fill='both', expand=True, pady=8)
        ttk.Label(frame, text='Esc：窗口内暂停。已发出的按压可能完成。\n异常自动暂停；游戏结束后请手动开局。').pack(anchor='w')
        root.bind('<Escape>', lambda _: self.pause())
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.poll)

    def emit(self, kind, value):
        self.events.put((kind, value))

    def start(self, mode):
        if self.worker and self.worker.is_alive():
            return
        try:
            coefficient, delay = float(self.coeff.get()), float(self.delay.get())
            if not (math.isfinite(coefficient) and .8 <= coefficient <= 2.5
                    and math.isfinite(delay) and .2 <= delay <= 10):
                raise ValueError
        except ValueError:
            messagebox.showerror('参数错误', '系数范围 0.8～2.5；等待时间范围 0.2～10 秒。')
            return
        self.save(coefficient, delay)
        self.stop.clear()
        self.count.set('本次已发起 0 跳')
        for b in self.buttons:
            b.configure(state='disabled')
        self.status.set('正在连接并读取画面……')
        def work():
            engine = Engine(self.emit, self.stop)
            try:
                if mode == 'preview':
                    model = engine.adb.connect()
                    scene = analyze(engine.stable())
                    self.emit('image', annotate(scene))
                    self.emit('status', f'{model} 识别成功，请核对红点位于下一平台顶面中心。')
                else:
                    engine.run(coefficient, delay, 3 if mode == 'calibrate' else 0)
            except InterruptedError:
                self.emit('status', '已暂停。')
            except Exception as exc:
                self.emit('status', f'已暂停：{exc}')
                self.log(str(exc))
            finally:
                self.emit('done', None)
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def save(self, coefficient=None, delay=None):
        if coefficient is not None:
            self.settings['coefficient'] = coefficient
        if delay is not None:
            self.settings['delay'] = delay
        temporary = ROOT/'settings.json.tmp'
        temporary.write_text(json.dumps(self.settings, indent=2), 'utf-8')
        temporary.replace(ROOT/'settings.json')

    def log(self, text):
        try:
            path = ROOT/'jumpbot.log'
            if path.exists() and path.stat().st_size > 2_000_000:
                path.replace(ROOT/'jumpbot.previous.log')
            with path.open('a', encoding='utf-8') as f:
                f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+text+'\n')
        except OSError:
            pass

    def pause(self):
        self.stop.set()
        self.status.set('正在暂停，等待已发出的操作结束……' if self.worker and self.worker.is_alive() else '已暂停。')

    def close(self):
        self.closing = True
        self.pause()
        self.root.withdraw()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'status':
                    self.status.set(value)
                    self.log(value)
                elif kind == 'count':
                    self.count.set(f'本次已发起 {value} 跳')
                elif kind == 'coefficient':
                    self.coeff.set(f'{value:.4f}')
                    self.save(coefficient=value)
                elif kind == 'image':
                    value = value.crop((0, int(value.height*.32), value.width, int(value.height*.83)))
                    value.thumbnail((460, 530))
                    self.photo = ImageTk.PhotoImage(value)
                    self.preview.configure(image=self.photo)
                elif kind == 'done':
                    for b in self.buttons:
                        b.configure(state='normal')
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self.poll)


if __name__ == '__main__':
    configure_dpi()
    mutex = None
    if hasattr(ctypes, 'windll'):
        kernel = ctypes.windll.kernel32
        kernel.CreateMutexW.restype = ctypes.c_void_p
        mutex = kernel.CreateMutexW(None, False, 'Local\\QQJumpBotHonor400')
        if kernel.GetLastError() == 183:
            kernel.CloseHandle(ctypes.c_void_p(mutex))
            messagebox.showinfo('已在运行', '助手窗口已经打开，请使用现有窗口。')
            raise SystemExit
    root = tk.Tk()
    App(root)
    try:
        root.mainloop()
    finally:
        if mutex:
            kernel.CloseHandle(ctypes.c_void_p(mutex))
