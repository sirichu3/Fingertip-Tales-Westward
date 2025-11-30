import os
import sys
import pygame
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

WIN_W, WIN_H = 1902, 1080
BG_COLOR = (0, 0, 0)

def to_surface(frame_bgr):
    import numpy as np
    frame_rgb = frame_bgr[:, :, ::-1]
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        font = pygame.font.SysFont('consolas', 22)
        running = True
        while running:
            screen.fill(BG_COLOR)
            t = font.render('缺少依赖: handpose/requirements.txt', True, (240,240,240))
            screen.blit(t, (40, 40))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
        pygame.quit()
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    img_path = os.path.join(PROJECT_DIR, 'vision_resources', 'hand.png')
    gate_img = pygame.image.load(img_path).convert_alpha()
    scale_w = int(WIN_W * 0.25)
    scale_h = int(gate_img.get_height() * (scale_w / max(1, gate_img.get_width())))
    gate_img = pygame.transform.smoothscale(gate_img, (scale_w, scale_h))
    gate_rect = gate_img.get_rect()
    gate_rect.center = (WIN_W // 2, WIN_H // 2)

    fade_alpha = 0
    state = 'gate'
    video_cap = None
    video_path = os.path.join(PROJECT_DIR, 'video', 'yun1.mp4')
    playback_speed = 3
    total_frames = 0
    current_frame = 0
    bottom_switched = False
    running = True
    do_zimu = False
    run_menu_after = False
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        ret, frame = cap.read()
        if not ret:
            screen.fill(BG_COLOR)
            pygame.display.flip()
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        cam = to_surface(frame)
        cam = pygame.transform.flip(cam, True, False)
        cam = pygame.transform.smoothscale(cam, (WIN_W, WIN_H))
        if state == 'play' and bottom_switched:
            pass
        else:
            screen.blit(cam, (0, 0))

        if state == 'gate':
            screen.blit(gate_img, gate_rect)
            hand_in = False
            pose_ok = False
            if result and result.multi_hand_landmarks:
                h = result.multi_hand_landmarks[0]
                def Lp(i):
                    lm = h.landmark[i]
                    return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
                THUMB_TIP = 4
                INDEX_TIP = 8
                MIDDLE_TIP = 12
                PINKY_TIP = 20
                INDEX_MCP = 5
                PINKY_MCP = 17
                WRIST = 0
                tt = Lp(THUMB_TIP)
                it = Lp(INDEX_TIP)
                mt = Lp(MIDDLE_TIP)
                pt = Lp(PINKY_TIP)
                imcp = Lp(INDEX_MCP)
                pmcp = Lp(PINKY_MCP)
                ws = Lp(WRIST)
                hand_pts = [Lp(i) for i in range(21)]
                hand_in = all(gate_rect.collidepoint(p) for p in hand_pts)
                import math
                hand_width = max(1, abs(imcp[0] - pmcp[0]))
                spread_tp = math.hypot(tt[0] - pt[0], tt[1] - pt[1])
                spread_im = math.hypot(it[0] - mt[0], it[1] - mt[1])
                pose_ok = (spread_tp >= 0.60 * hand_width) and (spread_im >= 0.15 * hand_width)
            if hand_in and pose_ok:
                try:
                    import cv2
                    video_cap = cv2.VideoCapture(video_path)
                    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    current_frame = 0
                    bottom_switched = False
                    state = 'play'
                except Exception:
                    do_zimu = True
                    running = False
        elif state == 'play':
            if video_cap is not None:
                ok, vframe = video_cap.read()
                if ok:
                    vsurf = to_surface(vframe)
                    vsurf = pygame.transform.smoothscale(vsurf, (WIN_W, WIN_H))
                    screen.blit(vsurf, (0, 0))
                    progressed = 1
                    extra = max(0, playback_speed - 1)
                    for _ in range(extra):
                        ok2, _ = video_cap.read()
                        if not ok2:
                            ok = False
                            break
                        progressed += 1
                    current_frame += progressed
                    mid = max(1, total_frames // 2) if total_frames > 0 else 30
                    if (not bottom_switched) and current_frame >= mid:
                        bottom_switched = True
                if not ok:
                    if video_cap is not None:
                        video_cap.release()
                    video_cap = None
                    do_zimu = True
                    running = False
            else:
                do_zimu = True
                running = False
        else:
            screen.fill(BG_COLOR)

        pygame.display.flip()

    cap.release()
    if video_cap is not None:
        video_cap.release()
    def run_title(screen_ref):
        clock2 = pygame.time.Clock()
        tp = os.path.join(PROJECT_DIR, 'vision_resources', 'title.png')
        try:
            title = pygame.image.load(tp).convert_alpha()
        except Exception:
            s = pygame.Surface((600, 180), pygame.SRCALPHA)
            title = s
        start = time.time()
        duration = 3.0
        fin = 0.5
        fout = 1.0
        r = True
        while r:
            clock2.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    r = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    r = False
            t = time.time() - start
            if t >= duration:
                break
            screen_ref.fill((0,0,0))
            rect = title.get_rect()
            rect.center = (WIN_W // 2, WIN_H // 2)
            a = 255
            if t < fin:
                a = int(255 * (t / fin))
            elif t > duration - fout:
                rem = duration - t
                a = int(255 * (rem / fout))
            ti = title.copy()
            ti.set_alpha(a)
            screen_ref.blit(ti, rect)
            pygame.display.flip()

    def run_camera_zimu(screen_ref):
        nonlocal run_menu_after
        clock2 = pygame.time.Clock()
        try:
            import cv2
            import mediapipe as mp
        except Exception:
            return
        cap2 = cv2.VideoCapture(0)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
        def to_surface2(frame_bgr):
            import numpy as np
            frame_rgb = frame_bgr[:, :, ::-1]
            h, w = frame_rgb.shape[:2]
            return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')
        def safe_load_bg(p):
            try:
                return pygame.image.load(p).convert_alpha()
            except Exception:
                s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                return s
        bg = safe_load_bg(os.path.join(PROJECT_DIR, 'vision_resources', 'zimubg.png'))
        def safe_load_img(p):
            try:
                return pygame.image.load(p).convert_alpha()
            except Exception:
                s = pygame.Surface((60, 24), pygame.SRCALPHA)
                s.fill((255,255,255))
                return s
        pointer = safe_load_img(os.path.join(PROJECT_DIR, 'vision_resources', 'pointer.png'))
        pointer_sel = safe_load_img(os.path.join(PROJECT_DIR, 'vision_resources', 'pointer_selected.png'))
        try:
            warming = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'warming.png')).convert_alpha()
        except Exception:
            warming = pygame.Surface((300, 90), pygame.SRCALPHA)
        pointer_norm_x = 0.5
        pointer_norm_target_x = pointer_norm_x
        pointer_vx = 0.0
        pointer_selected = False
        pinch_on_ratio = 0.35
        pinch_off_ratio = 0.50
        last_t = time.time()
        warming_show_until = 0.0
        mp_hands = mp.solutions.hands
        hands2 = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        zdir = os.path.join(PROJECT_DIR, 'vision_resources', 'zimu')
        try:
            names = sorted([n for n in os.listdir(zdir) if n.lower().endswith('.png')])
        except Exception:
            names = []
        def nat_key(n):
            import re
            m = re.search(r'(\D*)(\d+)', n)
            if m:
                return (m.group(1), int(m.group(2)))
            return (n, 0)
        names.sort(key=nat_key)
        images = []
        for n in names:
            p = os.path.join(zdir, n)
            try:
                img = pygame.image.load(p).convert_alpha()
                images.append(img)
            except Exception:
                pass
        if not images:
            s = pygame.Surface((200, 100))
            s.fill((0,0,0))
            images = [s]
        idx = 0
        first_loop_done = False
        duration_target = 5.0
        interval = max(3.0, min(10.0, duration_target))
        last_switch = time.time()
        r = True
        while r:
            clock2.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    r = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    r = False
            ret2, frame2 = cap2.read()
            if ret2:
                cam2 = to_surface2(frame2)
                cam2 = pygame.transform.flip(cam2, True, False)
                cam2 = pygame.transform.smoothscale(cam2, (WIN_W, WIN_H))
                screen_ref.blit(cam2, (0, 0))
                frame_rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                res = hands2.process(frame_rgb2)
                if res and res.multi_hand_landmarks:
                    h = res.multi_hand_landmarks[0]
                    def LM(lm):
                        return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
                    tt = LM(h.landmark[4])
                    it = LM(h.landmark[8])
                    tcmc = LM(h.landmark[1])
                    tmcp = LM(h.landmark[2])
                    tip_ip = LM(h.landmark[3])
                    imcp = LM(h.landmark[5])
                    mmcp = LM(h.landmark[9])
                    rmcp = LM(h.landmark[13])
                    pmcp = LM(h.landmark[17])
                    mt = LM(h.landmark[12])
                    rt = LM(h.landmark[16])
                    pt = LM(h.landmark[20])
                    ws = LM(h.landmark[0])
                    hw = max(1, abs(imcp[0] - pmcp[0]))
                    import math
                    dti = math.hypot(tt[0] - it[0], tt[1] - it[1])
                    pinch = (dti <= pinch_on_ratio * hw) if not pointer_selected else (dti <= pinch_off_ratio * hw)
                    if not pointer_selected and (dti <= pinch_on_ratio * hw):
                        pointer_selected = True
                    elif pointer_selected and (dti >= pinch_off_ratio * hw):
                        pointer_selected = False
                    if pointer_selected:
                        target_norm = max(0.2, min(0.8, it[0] / WIN_W))
                        pointer_norm_target_x = target_norm
                    idx_joints = [LM(h.landmark[i]) for i in (7,8)]
                    mid_joints = [LM(h.landmark[i]) for i in (11,12)]
                    ring_joints = [LM(h.landmark[i]) for i in (15,16)]
                    pink_joints = [LM(h.landmark[i]) for i in (19,20)]
                    ref_y = tmcp[1]
                    margin = int(0.04 * hw)
                    min_other_y = min(j[1] for j in (idx_joints + mid_joints + ring_joints + pink_joints))
                    others_below = (min_other_y > ref_y + margin)
                    ascend_margin = int(0.02 * hw)
                    thumb_ascend = (tcmc[1] > tmcp[1] + ascend_margin) and (tmcp[1] > tip_ip[1] + ascend_margin) and (tip_ip[1] > tt[1] + ascend_margin)
                    thumb_up = others_below and thumb_ascend
                    if thumb_up:
                        if not first_loop_done:
                            warming_show_until = time.time() + 1.5
                        else:
                            run_menu_after = True
                            r = False
            else:
                screen_ref.fill((0,0,0))
            now_t = time.time()
            dtp = max(1e-3, now_t - last_t)
            last_t = now_t
            K = 10.0
            C = 8.0
            ax = K * (pointer_norm_target_x - pointer_norm_x) - C * pointer_vx
            pointer_vx += ax * dtp
            pointer_norm_x += pointer_vx * dtp
            if pointer_norm_x < 0.2:
                pointer_norm_x = 0.2
                pointer_vx = 0.0
            elif pointer_norm_x > 0.8:
                pointer_norm_x = 0.8
                pointer_vx = 0.0
            px = int(WIN_W * pointer_norm_x)
            pr = (pointer_sel if pointer_selected else pointer).get_rect()
            pr.centerx = px
            pr.bottom = WIN_H - 4
            screen_ref.blit(pointer_sel if pointer_selected else pointer, pr)
            bgs = pygame.transform.smoothscale(bg, (WIN_W, WIN_H))
            screen_ref.blit(bgs, (0, 0))
            img = images[idx % len(images)]
            rect = img.get_rect()
            rect.center = (WIN_W // 2, WIN_H // 2)
            dt = time.time() - last_switch
            x = pointer_norm_x
            if x <= 0.5:
                t = (x - 0.2) / 0.3
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                interval = 3.0 + 2.0 * t
            else:
                t = (x - 0.5) / 0.3
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                interval = 5.0 + 5.0 * t
            if interval < 3.0:
                interval = 3.0
            elif interval > 10.0:
                interval = 10.0
            fade_in = 0.3
            fade_out = 0.3
            a = 255
            if dt < fade_in:
                a = int(255 * (dt / fade_in))
            elif dt > max(0.0, interval - fade_out):
                rem = max(0.0, interval - dt)
                a = int(255 * (rem / fade_out))
            blit_img = img.copy()
            blit_img.set_alpha(a)
            screen_ref.blit(blit_img, rect)
            if time.time() < warming_show_until:
                wr = warming.get_rect()
                wr.midtop = (WIN_W // 2, 88)
                screen_ref.blit(warming, wr)
            pygame.display.flip()
            if time.time() - last_switch >= interval:
                idx += 1
                last_switch = time.time()
                if (not first_loop_done) and (idx >= len(images)):
                    first_loop_done = True
        cap2.release()

    try:
        pygame.mixer.init()
        op = os.path.join(PROJECT_DIR, 'Audio', 'open.mp3')
        if os.path.exists(op):
            pygame.mixer.music.load(op)
            pygame.mixer.music.set_volume(0.8)
            pygame.mixer.music.play()
    except Exception:
        pass
    run_title(screen)
    run_camera_zimu(screen)
    if run_menu_after:
        try:
            import importlib.util
            mp = os.path.join(PROJECT_DIR, 'main_menu', 'menu.py')
            spec = importlib.util.spec_from_file_location('main_menu_menu', mp)
            menu_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(menu_mod)
            menu_mod.main()
        except Exception:
            pass
    pygame.quit()

if __name__ == '__main__':
    main()
