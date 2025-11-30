import os
import sys
import math
import pygame
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
WIN_W, WIN_H = 1902, 1080
SPRING_K = 22.0
DAMP_C = 11.0
ORIGIN_K = 24.0
ORIGIN_C = 12.0
ORIGIN_OFFSET_Y = 140
ORIGIN_MAX_SPEED_X = 900.0
ORIGIN_MAX_SPEED_Y = 1200.0
ORIGIN_MAX_DELTA = 28.0
ANGLE_GAIN = 1.15
ANGLE_GAIN_UPPER = 0.9
ARM_RANGE_GAIN = 3.80
ARM_RAISE_EXTRA_GAIN = 1.25
ARM_LOWER_EXTRA_GAIN = 1.15
ARM_BEND_INSET_PX = 12.0
ELBOW_RAISE_MARGIN = 24.0
ARM_RAISE_MODE_GAIN = 1.8
RAISE_POSE_TOL_PIX = 8
ROPE_LEN = 200
ROPE_STRETCH_RATIO_MAX = 0.05
HEAD_MIN_DEG = -38.0
HEAD_MAX_DEG = 38.0
SWAY_GAIN_X = 0.012
SWAY_MAX_DEG = 14.0
SWAY_K = 18.0
SWAY_DAMP = 8.0
WAIST_SWAY_RATIO = -0.6

sys.path.append(os.path.join(PROJECT_DIR, 'bone'))
from geom_puppet import GeomPuppet
from sprite_attach import SpriteManager

def to_surface(frame_bgr):
    import numpy as np
    frame_rgb = frame_bgr[:, :, ::-1]
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('consolas', 18)
    menu_music_started = False
    try:
        pygame.mixer.init()
        music_path = os.path.join(PROJECT_DIR, 'Audio', 'menu.mp3')
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.7)
            pygame.mixer.music.play(-1)
            menu_music_started = True
    except Exception:
        pass

    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        msg = [
            '缺少依赖：请先安装 handpose/requirements.txt',
            'pip install -r handpose/requirements.txt',
            f'ImportError: {e}',
        ]
        running = True
        while running:
            screen.fill((20,20,22))
            y = 40
            for t in msg:
                surf = font.render(t, True, (240,240,240))
                screen.blit(surf, (40, y))
                y += 28
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

    puppet = GeomPuppet()
    anchors_path = os.path.join(PROJECT_DIR, 'bone', 'anchors_sun.json')
    sm_sun = SpriteManager(puppet, anchors_path)
    sm_niu = SpriteManager(puppet, os.path.join(PROJECT_DIR, 'bone', 'anchors_niu.json'))
    bones = puppet.bones()

    origin_x = int(WIN_W * (2/3) * 0.5)
    origin_y = int(WIN_H * 0.55) + 100
    init_origin_x = origin_x
    init_origin_y = origin_y
    origin_vx = 0.0
    origin_vy = 0.0
    origin_offset_y = ORIGIN_OFFSET_Y
    G = 2200.0
    last_ox = origin_x
    last_oy = origin_y
    flip = False

    def world_to_screen(p):
        return (origin_x + (-(p.x) if flip else p.x), origin_y + p.y)

    def wrist_screen(b_lower):
        pos = b_lower.world_pos()
        tip = b_lower.tip_offset()
        wx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
        wy = origin_y + (pos.y + tip.y)
        return int(wx), int(wy)

    def leg_tip_screen(b_leg):
        pos = b_leg.world_pos()
        tip = b_leg.tip_offset()
        wx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
        wy = origin_y + (pos.y + tip.y)
        return int(wx), int(wy)

    def puppet_overlap_rect(rect):
        for b in bones:
            pos = b.world_pos()
            tip = b.tip_offset()
            jx = origin_x + (-(pos.x) if flip else pos.x)
            jy = origin_y + pos.y
            tx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
            ty = origin_y + (pos.y + tip.y)
            if rect.collidepoint(int(jx), int(jy)) or rect.collidepoint(int(tx), int(ty)):
                return True
        return False

    def screen_to_local(px, py):
        lx = -(px - origin_x) if flip else (px - origin_x)
        ly = py - origin_y
        return lx, ly

    def two_link_ik(shoulder_xy, target_xy, L1, L2, elbow_side=1):
        sx, sy = shoulder_xy
        tx, ty = target_xy
        dx, dy = tx - sx, ty - sy
        d = math.hypot(dx, dy)
        min_d = abs(L1 - L2) + 1e-6
        max_d = (L1 + L2) - 1e-6
        d_clamped = max(min_d, min(max_d, d))
        a = math.atan2(dy, dx)
        cos_phi = (L1*L1 + d_clamped*d_clamped - L2*L2) / (2*L1*d_clamped)
        cos_phi = max(-1.0, min(1.0, cos_phi))
        phi = math.acos(cos_phi)
        theta1 = a + elbow_side * phi
        ex = sx + L1 * math.cos(theta1)
        ey = sy + L1 * math.sin(theta1)
        theta2 = math.atan2(ty - ey, tx - ex)
        wx = ex + L2 * math.cos(theta2)
        wy = ey + L2 * math.sin(theta2)
        return (ex, ey), (wx, wy)

    # Right area assets
    free_mode_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'Free_Mode.png')).convert_alpha()
    load_save_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'Load_save.png')).convert_alpha()
    about_us_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'about_us.png')).convert_alpha()
    quit_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'quit.png')).convert_alpha()
    menu_bg_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'menu_background.png')).convert_alpha()
    menu_bg_raw = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'menu_background.png')).convert_alpha()
    def safe_load(path):
        try:
            return pygame.image.load(path).convert_alpha()
        except Exception:
            s = pygame.Surface((64, 64), pygame.SRCALPHA)
            return s
    sun_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'sun.png'))
    tieshan_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'tieshan.png'))
    tudi_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'tudi.png'))
    sun_icon.set_alpha(int(255*0.4))
    tieshan_icon.set_alpha(int(255*0.4))
    tudi_icon.set_alpha(int(255*0.4))
    fmw, fmh = free_mode_raw.get_width(), free_mode_raw.get_height()
    lsw, lsh = load_save_raw.get_width(), load_save_raw.get_height()
    auw, auh = about_us_raw.get_width(), about_us_raw.get_height()
    qw, qh = quit_raw.get_width(), quit_raw.get_height()
    free_mode = pygame.transform.smoothscale(free_mode_raw, (max(1, fmw//4), max(1, fmh//4)))
    load_save = pygame.transform.smoothscale(load_save_raw, (max(1, lsw//4), max(1, lsh//4)))
    about_us = pygame.transform.smoothscale(about_us_raw, (max(1, auw//4), max(1, auh//4)))
    quit_surf = pygame.transform.smoothscale(quit_raw, (max(1, qw//4), max(1, qh//4)))
    mouse_img = pygame.image.load(os.path.join(PROJECT_DIR, 'vision_resources', 'mouse.png')).convert_alpha()
    menu_bg = pygame.transform.smoothscale(menu_bg_raw, (WIN_W, WIN_H))
    menu_bg.set_alpha(int(255 * 0.8))
    def safe_load(path):
        try:
            return pygame.image.load(path).convert_alpha()
        except Exception:
            s = pygame.Surface((64, 64), pygame.SRCALPHA)
            return s
    sun_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'sun.png'))
    tieshan_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'tieshan.png'))
    tudi_icon = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'tudi.png'))
    sun_icon.set_alpha(int(255*0.4))
    tieshan_icon.set_alpha(int(255*0.4))
    tudi_icon.set_alpha(int(255*0.4))
    menu_bg = pygame.transform.smoothscale(menu_bg_raw, (WIN_W, WIN_H))
    menu_bg.set_alpha(int(255 * 0.8))
    gif_frames = []
    gif_dir = os.path.join(PROJECT_DIR, 'vision_resources', 'gif')
    for i in range(1, 26):
        gif_frames.append(pygame.image.load(os.path.join(gif_dir, f'{i}.png')).convert_alpha())

    left_w = int(WIN_W * 2 / 3)
    right_w = WIN_W - left_w
    right_rect = pygame.Rect(left_w, 0, right_w, WIN_H)
    center_x = left_w + right_w // 2 - 40
    SPACING = 90
    items = [free_mode, load_save, about_us, quit_surf]
    total_h = sum(s.get_height() for s in items) + SPACING * (len(items) - 1)
    start_y = (WIN_H - total_h) // 2
    free_rect = free_mode.get_rect()
    free_rect.centerx = center_x
    free_rect.top = start_y
    load_rect = load_save.get_rect()
    load_rect.centerx = center_x
    load_rect.top = free_rect.bottom + SPACING
    about_rect = about_us.get_rect()
    about_rect.centerx = center_x
    about_rect.top = load_rect.bottom + SPACING
    quit_rect = quit_surf.get_rect()
    quit_rect.centerx = center_x
    quit_rect.top = about_rect.bottom + SPACING
    HIT_BUFFER_PX = 40
    freemode_hit_rect = free_rect.inflate(HIT_BUFFER_PX * 2, HIT_BUFFER_PX * 2)
    loadsave_hit_rect = load_rect.inflate(HIT_BUFFER_PX * 2, HIT_BUFFER_PX * 2)
    unavailable_img = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'unavailable.png'))
    quit_hit_rect = quit_rect.inflate(HIT_BUFFER_PX * 2, HIT_BUFFER_PX * 2)
    aboutus_hit_rect = about_rect.inflate(HIT_BUFFER_PX * 2, HIT_BUFFER_PX * 2)
    dwell_target = None
    ding_sound = None
    try:
        from pygame import mixer as pg_mixer
        if not pg_mixer.get_init():
            pg_mixer.init()
        dp = os.path.join(PROJECT_DIR, 'Audio', 'ding.wav')
        if os.path.exists(dp):
            ding_sound = pg_mixer.Sound(dp)
            ding_sound.set_volume(0.9)
    except Exception:
        pass
    finger_inside = {'freemode': False, 'loadsave': False, 'aboutus': False, 'quit': False}
    puppet_inside = {'freemode': False, 'loadsave': False, 'aboutus': False, 'quit': False}

    # mapping state
    locked_active = False
    locked_mapping = {}
    locked_wrist_screen = None
    rope_len_map = {
        'left_hand': ROPE_LEN,
        'right_hand': ROPE_LEN,
        'left_leg': ROPE_LEN,
        'right_leg': ROPE_LEN,
        'head_joint': ROPE_LEN,
    }
    bone_state = {}
    for b in bones:
        bone_state[b.name] = {
            'theta': b.angle_deg,
            'omega': 0.0,
            'target': b.angle_deg,
            'rest': b.angle_deg,
        }
    raise_mode_left = False
    raise_mode_right = False
    body_sway_theta = 0.0
    body_sway_omega = 0.0
    waist_sway_theta = 0.0
    waist_sway_omega = 0.0
    dwell_start = None
    role_switch_start = None
    ROLE_SWITCH_SECONDS = 3.0
    role_progress = 0.0
    sun_rect_current = None
    DWELL_SECONDS = 3.0

    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ret, frame = cap.read()
        if not ret:
            screen.fill((18,18,20))
            pygame.display.flip()
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        cam = to_surface(frame)
        cam = pygame.transform.flip(cam, True, False)
        cam = pygame.transform.smoothscale(cam, (WIN_W, WIN_H))
        screen.blit(cam, (0, 0))
        screen.blit(menu_bg, (0, 0))
        ix = 250
        iy = 140
        mh = max(sun_icon.get_height(), tieshan_icon.get_height(), tudi_icon.get_height())
        r = sun_icon.get_rect()
        r.left = ix
        r.top = iy + (mh - sun_icon.get_height())
        screen.blit(sun_icon, r)
        sun_rect_current = r.copy()
        ix += r.width + 80
        r = tieshan_icon.get_rect()
        r.left = ix
        r.top = iy + (mh - tieshan_icon.get_height())
        screen.blit(tieshan_icon, r)
        ix += r.width + 80
        r = tudi_icon.get_rect()
        r.left = ix
        r.top = iy + (mh - tudi_icon.get_height())
        screen.blit(tudi_icon, r)
        screen.blit(free_mode, free_rect)
        screen.blit(load_save, load_rect)
        screen.blit(about_us, about_rect)
        screen.blit(quit_surf, quit_rect)

        # hand landmarks
        wrist_xy_screen = None
        thumb_tip_xy_screen = None
        index_tip_xy_screen = None
        middle_tip_xy_screen = None
        ring_tip_xy_screen = None
        pinky_tip_xy_screen = None

        if result and result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            def Lp(i):
                lm = hand.landmark[i]
                return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
            WRIST = 0
            THUMB_TIP = 4
            INDEX_TIP = 8
            MIDDLE_TIP = 12
            RING_TIP = 16
            PINKY_TIP = 20
            wrist_xy_screen = Lp(WRIST)
            thumb_tip_xy_screen = Lp(THUMB_TIP)
            index_tip_xy_screen = Lp(INDEX_TIP)
            middle_tip_xy_screen = Lp(MIDDLE_TIP)
            ring_tip_xy_screen = Lp(RING_TIP)
            pinky_tip_xy_screen = Lp(PINKY_TIP)

            blue_mode = False
            if index_tip_xy_screen and right_rect.collidepoint(index_tip_xy_screen):
                blue_mode = True
            out_screen = False
            if index_tip_xy_screen is None:
                out_screen = True
            else:
                ix, iy = index_tip_xy_screen
                if ix < 0 or ix >= WIN_W or iy < 0 or iy >= WIN_H:
                    out_screen = True
            recover_mode = blue_mode or out_screen
            if out_screen:
                locked_active = False
                locked_mapping = {}
                locked_wrist_screen = None

            if middle_tip_xy_screen is not None or recover_mode:
                if recover_mode:
                    txo = init_origin_x
                    tyo = init_origin_y
                else:
                    txo = middle_tip_xy_screen[0]
                    tyo = middle_tip_xy_screen[1] + origin_offset_y
                dt_o = 1.0 / max(1.0, clock.get_fps() or 30.0)
                ax_o = ORIGIN_K * (txo - origin_x) - ORIGIN_C * origin_vx
                Gy = 0.0
                ay_o = ORIGIN_K * (tyo - origin_y) - ORIGIN_C * origin_vy + Gy
                origin_vx += ax_o * dt_o
                origin_vy += ay_o * dt_o
                if origin_vx > ORIGIN_MAX_SPEED_X:
                    origin_vx = ORIGIN_MAX_SPEED_X
                elif origin_vx < -ORIGIN_MAX_SPEED_X:
                    origin_vx = -ORIGIN_MAX_SPEED_X
                if origin_vy > ORIGIN_MAX_SPEED_Y:
                    origin_vy = ORIGIN_MAX_SPEED_Y
                elif origin_vy < -ORIGIN_MAX_SPEED_Y:
                    origin_vy = -ORIGIN_MAX_SPEED_Y
                nx = origin_x + origin_vx * dt_o
                ny = origin_y + origin_vy * dt_o
                dx = nx - last_ox
                dy = ny - last_oy
                lim = ORIGIN_MAX_DELTA
                if dx > lim:
                    nx = last_ox + lim
                elif dx < -lim:
                    nx = last_ox - lim
                if dy > lim:
                    ny = last_oy + lim
                elif dy < -lim:
                    ny = last_oy - lim
                origin_x = nx
                origin_y = ny
                if origin_x < 0:
                    origin_x = 0
                elif origin_x > left_w:
                    origin_x = left_w
                if origin_y < 0:
                    origin_y = 0
                elif origin_y > WIN_H:
                    origin_y = WIN_H
                last_ox = origin_x
                last_oy = origin_y

            if ding_sound is not None:
                fi_fm = (index_tip_xy_screen is not None) and freemode_hit_rect.collidepoint(index_tip_xy_screen)
                fi_ls = (index_tip_xy_screen is not None) and loadsave_hit_rect.collidepoint(index_tip_xy_screen)
                fi_au = (index_tip_xy_screen is not None) and aboutus_hit_rect.collidepoint(index_tip_xy_screen)
                fi_qu = (index_tip_xy_screen is not None) and quit_hit_rect.collidepoint(index_tip_xy_screen)
                if fi_fm and not finger_inside['freemode']:
                    ding_sound.play()
                if fi_ls and not finger_inside['loadsave']:
                    ding_sound.play()
                if fi_au and not finger_inside['aboutus']:
                    ding_sound.play()
                if fi_qu and not finger_inside['quit']:
                    ding_sound.play()
                finger_inside['freemode'] = fi_fm
                finger_inside['loadsave'] = fi_ls
                finger_inside['aboutus'] = fi_au
                finger_inside['quit'] = fi_qu
                pi_fm = puppet_overlap_rect(freemode_hit_rect)
                pi_ls = puppet_overlap_rect(loadsave_hit_rect)
                pi_au = puppet_overlap_rect(aboutus_hit_rect)
                pi_qu = puppet_overlap_rect(quit_hit_rect)
                if pi_fm and not puppet_inside['freemode']:
                    ding_sound.play()
                if pi_ls and not puppet_inside['loadsave']:
                    ding_sound.play()
                if pi_au and not puppet_inside['aboutus']:
                    ding_sound.play()
                if pi_qu and not puppet_inside['quit']:
                    ding_sound.play()
                puppet_inside['freemode'] = pi_fm
                puppet_inside['loadsave'] = pi_ls
                puppet_inside['aboutus'] = pi_au
                puppet_inside['quit'] = pi_qu

            def drive_arm_by_rope(side: str, anchor_xy_screen: tuple, finger_id: int = None):
                if not anchor_xy_screen:
                    return
                ax, ay = anchor_xy_screen
                rlen = rope_len_map['left_hand'] if side == 'left' else rope_len_map['right_hand']
                rope_end_s = (ax, ay + rlen)
                tx, ty = screen_to_local(*rope_end_s)
                raise_this = False
                if finger_id in (8, 16) and middle_tip_xy_screen is not None:
                    fy = anchor_xy_screen[1]
                    my = middle_tip_xy_screen[1]
                    if fy <= my + RAISE_POSE_TOL_PIX:
                        raise_this = True
                if side == 'left':
                    raise_mode_left = bool(raise_this)
                else:
                    raise_mode_right = bool(raise_this)
                bu = next(b for b in bones if b.name == ('left_upper_arm' if side=='left' else 'right_upper_arm'))
                bl = next(b for b in bones if b.name == ('left_lower_arm' if side=='left' else 'right_lower_arm'))
                shoulder = bu.world_pos()
                sx, sy = shoulder.x, shoulder.y
                dx_rel, dy_rel = (tx - sx), (ty - sy)
                base_gain = ARM_RANGE_GAIN * (ARM_RAISE_MODE_GAIN if raise_this else 1.0)
                txg = sx + dx_rel * base_gain
                tyg = sy + dy_rel * base_gain
                if tyg < sy - ELBOW_RAISE_MARGIN:
                    dir_gain = ARM_RANGE_GAIN * ARM_RAISE_EXTRA_GAIN * (ARM_RAISE_MODE_GAIN if raise_this else 1.0)
                    txg = sx + dx_rel * dir_gain
                    tyg = sy + dy_rel * dir_gain
                elif tyg > sy + ELBOW_RAISE_MARGIN:
                    dir_gain = ARM_RANGE_GAIN * ARM_LOWER_EXTRA_GAIN
                    txg = sx + dx_rel * dir_gain
                    tyg = sy + dy_rel * dir_gain
                vxg, vyg = (txg - sx), (tyg - sy)
                d_g = math.hypot(vxg, vyg)
                if d_g > 1e-6:
                    inset = ARM_BEND_INSET_PX
                    new_d = max(0.0, d_g - inset)
                    sf = new_d / d_g
                    txg = sx + vxg * sf
                    tyg = sy + vyg * sf
                if not raise_this:
                    dx_off, dy_off = (txg - tx), (tyg - ty)
                    off = math.hypot(dx_off, dy_off)
                    limit = abs(rlen) * ROPE_STRETCH_RATIO_MAX
                    if off > limit and off > 1e-6:
                        k = limit / off
                        txg = tx + dx_off * k
                        tyg = ty + dy_off * k
                e1, w1 = two_link_ik((sx, sy), (txg, tyg), bu.length, bl.length, elbow_side=1)
                e2, w2 = two_link_ik((sx, sy), (txg, tyg), bu.length, bl.length, elbow_side=-1)
                elbow, wrist = (e1, w1) if e1[1] >= e2[1] else (e2, w2)
                def ang(ax, ay, bx, by):
                    vx, vy = bx - ax, by - ay
                    if vx == 0 and vy == 0:
                        return None
                    return math.degrees(math.atan2(-vx, vy))
                upper_global = ang(sx, sy, elbow[0], elbow[1])
                lower_global = ang(elbow[0], elbow[1], wrist[0], wrist[1])
                if upper_global is not None:
                    parent_angle = bu.parent.world_angle() if bu.parent else 0.0
                    bone_state[bu.name]['target'] = upper_global - parent_angle
                if lower_global is not None:
                    parent_angle = bl.parent.world_angle() if bl.parent else 0.0
                    bone_state[bl.name]['target'] = lower_global - parent_angle

            def drive_leg_by_rope(side: str, anchor_xy_screen: tuple):
                if not anchor_xy_screen:
                    return
                ax, ay = anchor_xy_screen
                rlen = rope_len_map['left_leg'] if side == 'left' else rope_len_map['right_leg']
                tx, ty = screen_to_local(ax, ay + rlen)
                bn = 'left_leg' if side == 'left' else 'right_leg'
                leg = next(b for b in bones if b.name == bn)
                hip = leg.parent.world_pos() if leg.parent else None
                sx, sy = hip.x, hip.y if hip else (0,0)
                vx, vy = tx - sx, ty - sy
                if not (abs(vx) < 1e-6 and abs(vy) < 1e-6):
                    glob = math.degrees(math.atan2(-vx, vy))
                    parent_angle = leg.parent.world_angle() if leg.parent else 0.0
                    bone_state[bn]['target'] = glob - parent_angle

            if not locked_active:
                tips_avail = {
                    THUMB_TIP: thumb_tip_xy_screen,
                    INDEX_TIP: index_tip_xy_screen,
                    MIDDLE_TIP: middle_tip_xy_screen,
                    RING_TIP: ring_tip_xy_screen,
                    PINKY_TIP: pinky_tip_xy_screen,
                }
                if all(v is not None for v in tips_avail.values()):
                    sorted_ids = sorted(tips_avail.keys(), key=lambda k: tips_avail[k][0])
                    target_order = ['left_leg', 'left_hand', 'head_joint', 'right_hand', 'right_leg']
                    locked_mapping = {fid: target_order[i] for i, fid in enumerate(sorted_ids)}
                    locked_wrist_screen = wrist_xy_screen
                    locked_active = True

            control_enabled = not recover_mode
            if locked_active and control_enabled:
                tips_by_id = {
                    THUMB_TIP: thumb_tip_xy_screen,
                    INDEX_TIP: index_tip_xy_screen,
                    MIDDLE_TIP: middle_tip_xy_screen,
                    RING_TIP: ring_tip_xy_screen,
                    PINKY_TIP: pinky_tip_xy_screen,
                }
                for fid, to_name in locked_mapping.items():
                    ft = tips_by_id.get(fid)
                    if ft is None:
                        continue
                    if to_name == 'left_hand':
                        drive_arm_by_rope('left', ft, fid)
                    elif to_name == 'right_hand':
                        drive_arm_by_rope('right', ft, fid)
                    elif to_name == 'left_leg':
                        drive_leg_by_rope('left', ft)
                    elif to_name == 'right_leg':
                        drive_leg_by_rope('right', ft)
                    elif to_name == 'head_joint':
                        b_head = next(b for b in bones if b.name == 'head')
                        pos = b_head.parent.world_pos() if b_head.parent else b_head.world_pos()
                        sx, sy = pos.x, pos.y
                        tx, ty = screen_to_local(ft[0], ft[1])
                        vx, vy = tx - sx, ty - sy
                        if not (abs(vx) < 1e-6 and abs(vy) < 1e-6):
                            glob = math.degrees(math.atan2(-vx, vy))
                            parent_angle = b_head.parent.world_angle() if b_head.parent else 0.0
                            rel = glob - parent_angle
                            if rel < HEAD_MIN_DEG:
                                rel = HEAD_MIN_DEG
                            elif rel > HEAD_MAX_DEG:
                                rel = HEAD_MAX_DEG
                            bone_state[b_head.name]['target'] = rel

            dt_phys = 1.0 / max(1.0, clock.get_fps() or 30.0)
            sway_target = origin_vx * SWAY_GAIN_X
            if sway_target < -SWAY_MAX_DEG:
                sway_target = -SWAY_MAX_DEG
            elif sway_target > SWAY_MAX_DEG:
                sway_target = SWAY_MAX_DEG
            tor_sway = -SWAY_K * (body_sway_theta - sway_target) - SWAY_DAMP * body_sway_omega
            body_sway_omega += tor_sway * dt_phys
            body_sway_theta += body_sway_omega * dt_phys
            waist_target = sway_target * WAIST_SWAY_RATIO
            tor_ws = -SWAY_K * (waist_sway_theta - waist_target) - SWAY_DAMP * waist_sway_omega
            waist_sway_omega += tor_ws * dt_phys
            waist_sway_theta += waist_sway_omega * dt_phys
            if 'body' in bone_state:
                bone_state['body']['target'] = bone_state['body']['rest'] + body_sway_theta
            if 'waist' in bone_state:
                bone_state['waist']['target'] = bone_state['waist']['rest'] + waist_sway_theta
            if recover_mode:
                for b in bones:
                    bs = bone_state[b.name]
                    bs['target'] = bs['rest']
                raise_mode_left = False
                raise_mode_right = False
            for b in bones:
                st = bone_state[b.name]
                tgt = st['target']
                is_left_raise = (b.name in ('left_upper_arm', 'left_lower_arm') and raise_mode_left)
                is_right_raise = (b.name in ('right_upper_arm', 'right_lower_arm') and raise_mode_right)
                if is_left_raise or is_right_raise:
                    if b.name in ('left_lower_arm', 'right_lower_arm'):
                        damp = DAMP_C * 1.0
                        spring_k = SPRING_K * 1.0
                    elif b.name in ('left_upper_arm', 'right_upper_arm'):
                        damp = DAMP_C * 0.5
                        spring_k = SPRING_K * 1.0
                    else:
                        damp = DAMP_C * 1.0
                        spring_k = SPRING_K * 1.0
                else:
                    if b.name in ('left_lower_arm', 'right_lower_arm'):
                        damp = DAMP_C * 0.8
                        spring_k = SPRING_K * 0.7
                    elif b.name in ('left_upper_arm', 'right_upper_arm'):
                        damp = DAMP_C * 0.4
                        spring_k = SPRING_K * 0.7
                    else:
                        damp = DAMP_C * 1.0
                        spring_k = SPRING_K * 1.0
                tor = -spring_k * (st['theta'] - tgt) - damp * st['omega']
                st['omega'] += tor * dt_phys
                st['theta'] += st['omega'] * dt_phys
                b.angle_deg = st['theta']

            if sun_rect_current and puppet_overlap_rect(sun_rect_current):
                if role_switch_start is None:
                    role_switch_start = time.time()
                else:
                    elapsed_role = time.time() - role_switch_start
                    role_progress = min(1.0, elapsed_role / ROLE_SWITCH_SECONDS)
            else:
                role_switch_start = None
                role_progress = 0.0

            if index_tip_xy_screen and right_rect.collidepoint(index_tip_xy_screen):
                idx_icon = mouse_img
                if loadsave_hit_rect.collidepoint(index_tip_xy_screen):
                    idx_icon = unavailable_img
                    dwell_start = None
                    dwell_target = None
                elif freemode_hit_rect.collidepoint(index_tip_xy_screen):
                    if dwell_target != 'freemode':
                        dwell_start = None
                    dwell_target = 'freemode'
                    if dwell_start is None:
                        dwell_start = time.time()
                    elapsed = time.time() - dwell_start
                    if elapsed <= DWELL_SECONDS:
                        fi = int(elapsed / DWELL_SECONDS * 25)
                        if fi < 0:
                            fi = 0
                        if fi > 24:
                            fi = 24
                        idx_icon = gif_frames[fi]
                    else:
                        idx_icon = gif_frames[24]
                elif aboutus_hit_rect.collidepoint(index_tip_xy_screen):
                    if dwell_target != 'aboutus':
                        dwell_start = None
                    dwell_target = 'aboutus'
                    if dwell_start is None:
                        dwell_start = time.time()
                    elapsed = time.time() - dwell_start
                    if elapsed <= DWELL_SECONDS:
                        fi = int(elapsed / DWELL_SECONDS * 25)
                        if fi < 0:
                            fi = 0
                        if fi > 24:
                            fi = 24
                        idx_icon = gif_frames[fi]
                    else:
                        idx_icon = gif_frames[24]
                elif quit_hit_rect.collidepoint(index_tip_xy_screen):
                    if dwell_target != 'quit':
                        dwell_start = None
                    dwell_target = 'quit'
                    if dwell_start is None:
                        dwell_start = time.time()
                    elapsed = time.time() - dwell_start
                    if elapsed <= DWELL_SECONDS:
                        fi = int(elapsed / DWELL_SECONDS * 25)
                        if fi < 0:
                            fi = 0
                        if fi > 24:
                            fi = 24
                        idx_icon = gif_frames[fi]
                    else:
                        idx_icon = gif_frames[24]
                else:
                    dwell_start = None
                    dwell_target = None
                ir = idx_icon.get_rect()
                ir.center = index_tip_xy_screen
                screen.blit(idx_icon, ir)
            else:
                dwell_start = None
                dwell_target = None

            if dwell_start is not None and (time.time() - dwell_start) >= DWELL_SECONDS:
                def run_demo_copy_on_existing_display(screen_ref):
                    import importlib.util
                    import pygame as pg
                    path = os.path.join(PROJECT_DIR, 'handpose2', 'demo copy.py')
                    orig_set_mode = pg.display.set_mode
                    orig_flip = pg.display.flip
                    orig_quit = pg.quit
                    scratch = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                    bg_img = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'freemodebg.png'))
                    def set_mode_passthrough(*args, **kwargs):
                        return scratch
                    def flip_overlay():
                        if bg_img:
                            try:
                                bg_scaled = pygame.transform.smoothscale(bg_img, (WIN_W, WIN_H))
                                screen_ref.blit(bg_scaled, (0, 0))
                            except Exception:
                                pass
                        try:
                            screen_ref.blit(scratch, (0, 0))
                        except Exception:
                            pass
                        orig_flip()
                    def noop_quit(*args, **kwargs):
                        return None
                    pg.display.set_mode = set_mode_passthrough
                    pg.display.flip = flip_overlay
                    pg.quit = noop_quit
                    try:
                        spec = importlib.util.spec_from_file_location('handpose2_demo_copy', path)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        mod.main()
                    finally:
                        pg.display.set_mode = orig_set_mode
                        pg.display.flip = orig_flip
                        pg.quit = orig_quit
                def run_aboutus_pull_on_existing_display(screen_ref):
                    try:
                        import cv2
                        import numpy as np
                    except Exception:
                        return
                    vp_pull = os.path.join(PROJECT_DIR, 'video', 'pull.mp4')
                    vp_trad = os.path.join(PROJECT_DIR, 'video', 'traditionpiying.mp4')
                    cap_pull = cv2.VideoCapture(vp_pull)
                    cap_trad = cv2.VideoCapture(vp_trad)
                    cap_cam = cv2.VideoCapture(0)
                    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
                    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
                    zimu_started = False
                    try:
                        pygame.mixer.init()
                        zimu_path = os.path.join(PROJECT_DIR, 'Audio', 'zimu.mp3')
                        if os.path.exists(zimu_path):
                            pygame.mixer.music.load(zimu_path)
                            pygame.mixer.music.set_volume(0.7)
                            pygame.mixer.music.play(-1)
                            zimu_started = True
                    except Exception:
                        pass
                    def read_frame(cap):
                        if cap is None:
                            return None
                        ok, fr = cap.read()
                        if not ok:
                            try:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ok, fr = cap.read()
                            except Exception:
                                return None
                        return fr if ok else None
                    first_pull = read_frame(cap_pull)
                    first_trad = read_frame(cap_trad)
                    def to_surf(fr):
                        rgb = fr[:, :, ::-1]
                        h, w = rgb.shape[:2]
                        return pygame.image.frombuffer(rgb.tobytes(), (w, h), 'RGB')
                    pull_h = WIN_H if first_pull is None else int(WIN_W * (first_pull.shape[0] / max(1, first_pull.shape[1])))
                    trad_target_w = WIN_W // 2
                    trad_h = WIN_H // 2 if first_trad is None else int(trad_target_w * (first_trad.shape[0] / max(1, first_trad.shape[1])))
                    about_team = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'aboutteam.png'))
                    about_piying = safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'aboutpiying.png'))
                    about_team = pygame.transform.smoothscale(about_team, (WIN_W, int(about_team.get_height() * WIN_W / max(1, about_team.get_width()))))
                    about_piying = pygame.transform.smoothscale(about_piying, (WIN_W, int(about_piying.get_height() * WIN_W / max(1, about_piying.get_width()))))
                    spacer = 30
                    y0 = 0
                    y1 = y0 + pull_h
                    y2 = y1 + about_team.get_height()
                    y3 = y2 + spacer
                    y4 = y3 + trad_h
                    y5 = y4 + spacer
                    total_h = y5 + about_piying.get_height()
                    scroll_y = 0.0
                    auto_scroll = False
                    trad_done = False
                    AUTO_STEP = 6
                    mp_hands = mp.solutions.hands
                    hands2 = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    last_center = None
                    fold_active = False
                    fold_seen_once = False
                    last_dy = 0.0
                    clock2 = pygame.time.Clock()
                    running2 = True
                    while running2:
                        clock2.tick(30)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running2 = False
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                running2 = False
                            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                                if (not trad_done) and (not auto_scroll):
                                    trad_done = True
                                    auto_scroll = True
                        # camera + gesture
                        screen_ref.fill((0,0,0))
                        ok_cam, fr_cam = cap_cam.read()
                        if ok_cam:
                            fr_rgb = cv2.cvtColor(fr_cam, cv2.COLOR_BGR2RGB)
                            res = hands2.process(fr_rgb)
                            fold_active = False
                            center_now = None
                            if res and res.multi_hand_landmarks:
                                h = res.multi_hand_landmarks[0]
                                def LM(i):
                                    lm = h.landmark[i]
                                    return (int(lm.x * WIN_W), int(lm.y * WIN_H))
                                imcp = LM(5)
                                pmcp = LM(17)
                                hw = max(1, abs(imcp[0] - pmcp[0]))
                                margin = int(0.04 * hw)
                                def folded(mcp_i, tip_i):
                                    m = LM(mcp_i)[1]
                                    t = LM(tip_i)[1]
                                    return t > m + margin
                                f_index = folded(5, 8)
                                f_middle = folded(9, 12)
                                f_ring = folded(13, 16)
                                f_pinky = folded(17, 20)
                                fold_active = f_index and f_middle and f_ring and f_pinky
                                center_now = LM(0)  # wrist
                            if fold_active:
                                auto_scroll = False
                                fold_seen_once = True
                                if last_center and center_now:
                                    dy = center_now[1] - last_center[1]
                                    MANUAL_GAIN = 0.6
                                    scroll_y += -dy * MANUAL_GAIN
                                    last_dy = dy
                            else:
                                auto_scroll = True if fold_seen_once else False
                                last_dy = 0.0
                            if center_now:
                                last_center = center_now
                        # clamp scroll within bounds: top at pull top, bottom at about_piying bottom
                        min_scroll = WIN_H - total_h
                        max_scroll = 0.0
                        if scroll_y < min_scroll:
                            scroll_y = min_scroll
                        elif scroll_y > max_scroll:
                            scroll_y = max_scroll
                        # pause auto scroll while traditionpiying is playing
                        trad_top_pre = int(y3 + scroll_y)
                        trad_bottom_pre = trad_top_pre + trad_h
                        trad_playing = (not trad_done) and (trad_top_pre >= 0 and trad_bottom_pre <= WIN_H)
                        if auto_scroll and (not trad_playing):
                            prev_sy = scroll_y
                            scroll_y += -AUTO_STEP
                            if scroll_y < min_scroll:
                                scroll_y = min_scroll
                        if fold_active and (scroll_y <= min_scroll) and (last_dy > 0.0):
                            try:
                                if zimu_started:
                                    pygame.mixer.music.stop()
                            except Exception:
                                pass
                            try:
                                main()
                            except Exception:
                                pass
                            running2 = False
                        # draw items
                        # pull video always looping
                        fr_pull = read_frame(cap_pull)
                        if fr_pull is not None:
                            sf = to_surf(fr_pull)
                            sf = pygame.transform.smoothscale(sf, (WIN_W, pull_h))
                            rect = sf.get_rect()
                            rect.topleft = (0, y0 + scroll_y)
                            screen_ref.blit(sf, rect)
                        # aboutteam image
                        at_rect = about_team.get_rect()
                        at_rect.topleft = (0, y1 + scroll_y)
                        screen_ref.blit(about_team, at_rect)
                        # spacer area y2
                        # tradition video play/pause based on full visibility
                        trad_rect = pygame.Rect(0, int(y3 + scroll_y), trad_target_w, trad_h)
                        trad_rect.centerx = WIN_W // 2
                        fully_inside = (trad_rect.top >= 0 and trad_rect.bottom <= WIN_H)
                        fully_outside = (trad_rect.bottom <= 0 or trad_rect.top >= WIN_H)
                        if (not trad_done) and fully_inside:
                            fr_trad = read_frame(cap_trad)
                            if fr_trad is not None:
                                ts = to_surf(fr_trad)
                                ts = pygame.transform.smoothscale(ts, (trad_target_w, trad_h))
                                screen_ref.blit(ts, trad_rect)
                            else:
                                trad_done = True
                        else:
                            # paused: do not advance
                            pass
                        # spacer y4
                        ap_rect = about_piying.get_rect()
                        ap_rect.topleft = (0, y5 + scroll_y)
                        screen_ref.blit(about_piying, ap_rect)
                        pygame.display.flip()
                        # exit to menu at bottom in auto mode
                        if scroll_y <= min_scroll and auto_scroll and trad_done:
                            try:
                                if zimu_started:
                                    pygame.mixer.music.stop()
                            except Exception:
                                pass
                            try:
                                main()
                            except Exception:
                                pass
                            running2 = False
                    if cap_pull is not None:
                        cap_pull.release()
                    if cap_trad is not None:
                        cap_trad.release()
                    if cap_cam is not None:
                        cap_cam.release()
                    try:
                        if zimu_started:
                            pygame.mixer.music.stop()
                    except Exception:
                        pass
                    # no camera to release in auto mode
                if dwell_target == 'freemode':
                    try:
                        if menu_music_started:
                            pygame.mixer.music.stop()
                    except Exception:
                        pass
                    run_demo_copy_on_existing_display(screen)
                    dwell_start = None
                    dwell_target = None
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
                    except Exception:
                        pass
                    try:
                        music_path = os.path.join(PROJECT_DIR, 'Audio', 'menu.mp3')
                        if os.path.exists(music_path):
                            pygame.mixer.music.load(music_path)
                            pygame.mixer.music.set_volume(0.7)
                            pygame.mixer.music.play(-1)
                            menu_music_started = True
                    except Exception:
                        pass
                elif dwell_target == 'aboutus':
                    try:
                        if menu_music_started:
                            pygame.mixer.music.stop()
                    except Exception:
                        pass
                    run_aboutus_pull_on_existing_display(screen)
                    dwell_start = None
                    dwell_target = None
                    try:
                        cap.release()
                    except Exception:
                        pass
                    try:
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)
                    except Exception:
                        pass
                elif dwell_target == 'quit':
                    try:
                        if menu_music_started:
                            pygame.mixer.music.stop()
                    except Exception:
                        pass
                    running = False

        

        surf_sun = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        surf_niu = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        sm_sun.draw(surf_sun, origin_x, origin_y, show_bones=False, flip=flip, show_endpoints=False)
        sm_niu.draw(surf_niu, origin_x, origin_y, show_bones=False, flip=flip, show_endpoints=False)
        sa = int(255 * max(0.0, 1.0 - role_progress))
        na = int(255 * min(1.0, role_progress))
        surf_sun.set_alpha(sa)
        surf_niu.set_alpha(na)
        screen.blit(surf_sun, (0, 0))
        screen.blit(surf_niu, (0, 0))
        pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()
