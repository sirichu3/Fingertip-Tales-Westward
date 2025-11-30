import os
import sys
import math
import time
import json
import base64
import io
import random

import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Import puppet geometry from bone demo
sys.path.append(os.path.join(PROJECT_DIR, 'bone'))
from geom_puppet import GeomPuppet
from sprite_attach import SpriteManager
# Import IK helper for two-link arm
sys.path.append(PROJECT_DIR)
# 本地实现两段臂IK，避免引入不必要依赖
def two_link_ik(shoulder_xy, target_xy, L1, L2, elbow_side=1):
    sx, sy = shoulder_xy
    tx, ty = target_xy
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    # 夹取到可达范围
    min_d = abs(L1 - L2) + 1e-6
    max_d = (L1 + L2) - 1e-6
    d_clamped = max(min_d, min(max_d, d))
    # 方向角（屏幕坐标：y向下为正）
    a = math.atan2(dy, dx)
    # 肘部相对肩的偏转角
    cos_phi = (L1*L1 + d_clamped*d_clamped - L2*L2) / (2*L1*d_clamped)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phi = math.acos(cos_phi)
    theta1 = a + elbow_side * phi
    ex = sx + L1 * math.cos(theta1)
    ey = sy + L1 * math.sin(theta1)
    # 手腕沿肘→目标的方向前进L2
    theta2 = math.atan2(ty - ey, tx - ex)
    wx = ex + L2 * math.cos(theta2)
    wy = ey + L2 * math.sin(theta2)
    return (ex, ey), (wx, wy)


WIN_W, WIN_H = 1920, 1080
BG_COLOR = (18, 18, 20)
GROUND_Y_RATIO = 0.90  # 地面位置
GROUND_COLOR = (80, 80, 85)
TEXT_COLOR = (240, 240, 240)

BONE_COLOR = (0, 200, 255)
SEL_COLOR = (255, 180, 0)

# Simple spring-damper parameters for physical swing
SPRING_K = 22.0
DAMP_C = 11.0

# Origin (puppet root) spring for hanging under the hand
ORIGIN_K = 24.0
ORIGIN_C = 12.0
ORIGIN_OFFSET_Y = 140  # distance below wrist
ORIGIN_MAX_SPEED_X = 5000.0
ORIGIN_MAX_SPEED_Y = 12000.0
ORIGIN_MAX_DELTA = 50.0
ARM_RAISE_MODE_GAIN = 1.8
RAISE_POSE_TOL_PIX = 8
ROPE_STRETCH_RATIO_MAX = 0.05
MATCH_THRESH_BODY = 16.0
MATCH_THRESH_WAIST = 30.0
MATCH_THRESH_OTHERS = 24.0

# Sensitivity gains
ANGLE_GAIN = 1.15  # finger→lower-arm响应增益
ANGLE_GAIN_UPPER = 0.9  # finger→upper-arm响应增益
# 扩大两臂运动范围的增益（在不改变手指运动范围的前提下）
ARM_RANGE_GAIN = 3.80
# 方向性增益：抬起/下伸更容易
ARM_RAISE_EXTRA_GAIN = 1.25
ARM_LOWER_EXTRA_GAIN = 1.15
# 弯曲偏好：目标向肩内收以增加肘部弯曲（像素）
ARM_BEND_INSET_PX = 12.0
ELBOW_RAISE_MARGIN = 24.0  # 目标高于肩部多少像素算“抬起”
SWAY_ACC_X = 0.24  # origin加速度到角度的映射（水平）
SWAY_ACC_Y = 0.12  # origin加速度到角度的映射（竖直）
SWAY_LIMIT_DEG = 48.0
BODY_TENSION_GAIN = 0.32  # 腿部绳子上拉对躯干倾斜的增益（deg/px）
HIP_TENSION_GAIN = BODY_TENSION_GAIN * 2.0  # 髋部相对躯干的上拉增益为两倍

# Puppet string shorten factor (towards puppet end)
STRING_TRIM = 0.10

# Rope parameters: vertical hanging length from fingertip anchor to wrist target
ROPE_LEN = 200
# Per-target rope length calibration map (defaults to ROPE_LEN)
ROPE_LEN_MAP_DEFAULT = {
    'left_hand': ROPE_LEN,
    'right_hand': ROPE_LEN,
    'left_leg': ROPE_LEN,
    'right_leg': ROPE_LEN,
    'head_joint': ROPE_LEN,
}


def to_pygame_surface(frame_bgr):
    import numpy as np
    frame_rgb = frame_bgr[:, :, ::-1]
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')


def finger_angle_deg(base_xy, tip_xy):
    bx, by = base_xy
    tx, ty = tip_xy
    vx = tx - bx
    vy = ty - by
    if vx == 0 and vy == 0:
        return None
    # Global angle so that local +Y points to (vx, vy)
    return math.degrees(math.atan2(-vx, vy))


def _load_shadow_surface():
    path = os.path.join(PROJECT_DIR, 'vision_resources', 'inside_cave.png')
    try:
        surf = pygame.image.load(path).convert_alpha()
        surf = pygame.transform.smoothscale(surf, (WIN_W, WIN_H))
        surf.set_alpha(204)
        return surf
    except Exception:
        return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Handpose Demo — 摄像头底层/Shadow/人偶叠加 + 物理摆动')
    clock = pygame.time.Clock()

    # Try to import cv2 + mediapipe; fail gracefully if missing
    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        font = pygame.font.SysFont('consolas', 18)
        msg = [
            '缺少依赖：请先安装 handpose/requirements.txt',
            'pip install -r handpose/requirements.txt',
            f'ImportError: {e}',
        ]
        running = True
        while running:
            screen.fill(BG_COLOR)
            y = 40
            for line in msg:
                surf = font.render(line, True, TEXT_COLOR)
                screen.blit(surf, (40, y))
                y += 28
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
        pygame.quit()
        return

    # Camera init
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    puppet = GeomPuppet()
    bones = puppet.bones()
    # Sprite manager for role 'niu' with flip/non-flip toggle
    anchors_is_flip = False
    anchors_path = os.path.join(PROJECT_DIR, 'bone', 'anchors_niu_flip.json' if anchors_is_flip else 'anchors_niu.json')
    sm = SpriteManager(puppet, anchors_path)
    # dynamic origin state (hang below wrist)
    origin_x, origin_y = int(WIN_W * 0.75), int(WIN_H * 0.50)
    origin_vy = 0.0
    G = 2200.0  # 重力加速度（px/s^2）
    fall_side = 0  # 倒向方向：-1 左，+1 右，0 未确定
    # 肘部随机化状态：左右臂基础方向分别为左(+1)/右(-1)
    elbow_side_left = 1
    elbow_side_right = -1

    # 计分系统：加载目标姿态（pose/test.json），创建虚影骨骼并随机放置
    score = 0
    ghost = GeomPuppet()
    ghost_bones = ghost.bones()
    ghost_origin = [int(WIN_W * 0.75) - 100, int(WIN_H * 0.35)]
    def load_ghost_pose():
        path = os.path.join(PROJECT_DIR, 'pose', 'test.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return False
        ang = data.get('angles', {})
        for b in ghost_bones:
            if b.name in ang:
                try:
                    b.angle_deg = float(ang[b.name])
                except Exception:
                    pass
        return True
    def relocate_ghost():
        ghost_origin[0] = int(WIN_W * 0.75) - 100
        ghost_origin[1] = int(WIN_H * 0.35)
    load_ghost_pose()
    relocate_ghost()
    origin_vx = origin_vy = 0.0
    origin_offset_y = ORIGIN_OFFSET_Y
    flip = False
    # Neutral calibration for finger angles (open hand → ±45° upper, 0° lower)
    DEFAULT_NEUTRAL_DEG = 180.0
    thumb_neutral_deg = DEFAULT_NEUTRAL_DEG
    pinky_neutral_deg = DEFAULT_NEUTRAL_DEG
    calibrate_request = False
    # Rope calibration state: per-target rope lengths
    rope_len_map = dict(ROPE_LEN_MAP_DEFAULT)
    calibrate_rope_request = False

    # per-bone physical states: angle follows target via spring-damper
    bone_state = {}
    for b in bones:
        bone_state[b.name] = {
            'theta': b.angle_deg,
            'omega': 0.0,
            'target': b.angle_deg,
            'rest': b.angle_deg,
        }

    shadow_surface = _load_shadow_surface()
    try:
        tieshan_path = os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'tieshangongzhu.png')
        tieshan_img = pygame.image.load(tieshan_path).convert_alpha()
        # 保持原尺寸，不进行缩放
    except Exception:
        tieshan_img = None

    # Shake detection state
    last_wrist = None
    last_wrist_vx = 0.0
    last_shake_sample_time = 0.0
    prev_ovx, prev_ovy = 0.0, 0.0
    last_ox, last_oy = origin_x, origin_y
    last_flip_time = 0.0
    FLIP_COOLDOWN = 0.6
    # 使用横向加速度判定甩动（像素/秒^2）
    SHAKE_ACC_THRESH = 6000.0

    font = pygame.font.SysFont('consolas', 18)

    # 手-人偶链接锁定：一旦建立，手不离开摄像头就不改变映射
    locked_active = False
    locked_mapping = {}  # {finger_tip_index:int -> target_name:str}
    locked_wrist_screen = None
    # 一帧的“中立展示”请求：按下 n 后当帧不执行驱动，直接展示中立姿态
    neutral_frame_request = False
    # 躯干-髋部二连杆的腰部摆动状态（物理式阻尼耦合）
    trunk_waist_drive = 0.0
    trunk_waist_vel = 0.0
    # 骨架显示开关（Space 切换）
    show_bones = False
    raise_mode_left = False
    raise_mode_right = False

    def load_collision_object(name):
        try:
            p = os.path.join(PROJECT_DIR, 'collision_volume', name)
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sub = data.get('image')
            sc = float(data.get('scale', 1.0))
            shapes = data.get('shapes', []) or []
            ip = os.path.join(PROJECT_DIR, 'vision_resources', sub)
            img = pygame.image.load(ip).convert_alpha()
            return {'image': img, 'scale': sc, 'shapes': shapes, 'sub': sub}
        except Exception:
            return None

    running = True
    obj = None
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    puppet.reset_pose()
                elif event.key == pygame.K_n:
                    # 立即重置人偶到“45°向下张开，双腿直立”姿态，并重置映射
                    puppet.reset_pose()
                    # 头部与躯干（以及腰部）也回到原始中立角度
                    try:
                        puppet.head.angle_deg = 0.0
                        puppet.body.angle_deg = 0.0
                        puppet.waist.angle_deg = 0.0
                    except Exception:
                        pass
                    # 同步物理状态为当前角度，避免旧目标造成延迟
                    for b in bones:
                        st = bone_state[b.name]
                        st['theta'] = b.angle_deg
                        st['target'] = b.angle_deg
                        st['omega'] = 0.0
                        st['rest'] = b.angle_deg
                    # 清空锁定，下一帧按五指从左到右重新建立映射
                    locked_active = False
                    locked_mapping = {}
                    locked_wrist_screen = None
                    # 本帧直接展示中立姿态，不进行驱动
                    neutral_frame_request = True
                    # 触发绳长校准（在重新锁定后执行）
                    calibrate_rope_request = True
                elif event.key == pygame.K_SPACE:
                    # 切换原始人偶骨骼显示
                    show_bones = not show_bones

        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            screen.fill(BG_COLOR)
            surf = font.render('摄像头未就绪', True, TEXT_COLOR)
            screen.blit(surf, (40, 40))
            pygame.display.flip()
            continue

        # MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        thumb_deg = pinky_deg = middle_deg = None
        wrist_xy = None
        thumb_tip_xy = pinky_tip_xy = middle_tip_xy = None
        # 屏幕坐标预设为 None，避免未检测到手时未定义
        wrist_xy_screen = None
        thumb_tip_xy_screen = None
        index_tip_xy_screen = None
        middle_tip_xy_screen = None
        ring_tip_xy_screen = None
        pinky_tip_xy_screen = None
        if result.multi_hand_landmarks:
            hands_list = result.multi_hand_landmarks
            # 选择手：若已锁定则选择距离上次锁定腕点最近的那只，否则取第一只
            WRIST = 0
            def LM_from(h, idx):
                lm = h.landmark[idx]
                return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
            if locked_active and locked_wrist_screen is not None:
                best = None
                best_d2 = 1e9
                for h in hands_list:
                    ws = LM_from(h, WRIST)
                    dx = ws[0] - locked_wrist_screen[0]
                    dy = ws[1] - locked_wrist_screen[1]
                    d2 = dx*dx + dy*dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best = h
                hand = best or hands_list[0]
            else:
                hand = hands_list[0]
            h, w = frame.shape[:2]

            # landmark indices
            THUMB_MCP, THUMB_TIP = 2, 4
            PINKY_MCP, PINKY_TIP = 17, 20
            MIDDLE_MCP, MIDDLE_TIP = 9, 12
            INDEX_TIP, RING_TIP = 8, 16

            def L(lm):
                return (int(lm.x * w), int(lm.y * h))

            wrist_xy = L(hand.landmark[WRIST])
            thumb_tip_xy = L(hand.landmark[THUMB_TIP])
            pinky_tip_xy = L(hand.landmark[PINKY_TIP])
            middle_tip_xy = L(hand.landmark[MIDDLE_TIP])
            index_tip_xy = L(hand.landmark[INDEX_TIP])
            ring_tip_xy = L(hand.landmark[RING_TIP])
            # 屏幕坐标（随摄像头层水平镜像）
            def LM(lm):
                return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
            wrist_xy_screen = LM(hand.landmark[WRIST])
            thumb_tip_xy_screen = LM(hand.landmark[THUMB_TIP])
            pinky_tip_xy_screen = LM(hand.landmark[PINKY_TIP])
            middle_tip_xy_screen = LM(hand.landmark[MIDDLE_TIP])
            index_tip_xy_screen = LM(hand.landmark[INDEX_TIP])
            ring_tip_xy_screen = LM(hand.landmark[RING_TIP])
            thumb_deg = finger_angle_deg(L(hand.landmark[THUMB_MCP]), L(hand.landmark[THUMB_TIP]))
            pinky_deg = finger_angle_deg(L(hand.landmark[PINKY_MCP]), L(hand.landmark[PINKY_TIP]))
            middle_deg = finger_angle_deg(L(hand.landmark[MIDDLE_MCP]), L(hand.landmark[MIDDLE_TIP]))

            # Apply neutral calibration when requested
            if calibrate_request:
                if thumb_deg is not None:
                    thumb_neutral_deg = thumb_deg
                if pinky_deg is not None:
                    pinky_neutral_deg = pinky_deg
                calibrate_request = False

            # Shake detection: horizontal acceleration of wrist
            now = time.time()
            if last_wrist is not None:
                dt = max(1e-3, (now - last_shake_sample_time) if last_shake_sample_time > 0 else (1.0/60.0))
                dx = wrist_xy[0] - last_wrist[0]
                vx = dx / dt
                ax_wrist = (vx - last_wrist_vx) / dt
                # 需要加速度超过阈值且速度发生左右切换（更像甩动）
                if abs(ax_wrist) > SHAKE_ACC_THRESH and (vx * last_wrist_vx < 0) and (now - last_flip_time) > FLIP_COOLDOWN:
                    last_flip_time = now
                    anchors_is_flip = not anchors_is_flip
                    anchors_path = os.path.join(PROJECT_DIR, 'bone', 'anchors_niu_flip.json' if anchors_is_flip else 'anchors_niu.json')
                    sm = SpriteManager(puppet, anchors_path)
                    calibrate_request = True
                last_wrist_vx = vx
                last_shake_sample_time = now
            last_wrist = wrist_xy

            # Screen→local坐标转换（考虑origin与flip）
            def screen_to_local(px, py):
                lx = -(px - origin_x) if flip else (px - origin_x)
                ly = py - origin_y
                return lx, ly

            # 用垂直绳子的末端作为手腕目标，通过两段臂IK驱动上下臂
            def drive_arm_by_rope(side: str, anchor_xy_screen: tuple, finger_id: int = None):
                nonlocal elbow_side_left, elbow_side_right, raise_mode_left, raise_mode_right
                if not anchor_xy_screen:
                    return
                ax, ay = anchor_xy_screen
                # 依据侧别选择校准后的绳长
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
                if side == 'left':
                    bu = next(b for b in bones if b.name == 'left_upper_arm')
                    bl = next(b for b in bones if b.name == 'left_lower_arm')
                    shoulder = bu.world_pos()
                    sx, sy = shoulder.x, shoulder.y
                    # 将目标相对肩部的位移按增益放大，扩大两臂的运动范围
                    dx_rel, dy_rel = (tx - sx), (ty - sy)
                    if raise_this:
                        base_gain = ARM_RANGE_GAIN * ARM_RAISE_MODE_GAIN
                        txg = sx + dx_rel * base_gain
                        tyg = sy + dy_rel * base_gain
                        if tyg < sy - ELBOW_RAISE_MARGIN:
                            dir_gain = ARM_RANGE_GAIN * ARM_RAISE_EXTRA_GAIN * ARM_RAISE_MODE_GAIN
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
                        dx_off, dy_off = (txg - tx), (tyg - ty)
                        off = math.hypot(dx_off, dy_off)
                        limit = abs(rlen) * ROPE_STRETCH_RATIO_MAX
                        if off > limit and off > 1e-6:
                            k = limit / off
                            txg = tx + dx_off * k
                            tyg = ty + dy_off * k
                    else:
                        txg = tx
                        tyg = ty
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
                elif side == 'right':
                    bu = next(b for b in bones if b.name == 'right_upper_arm')
                    bl = next(b for b in bones if b.name == 'right_lower_arm')
                    shoulder = bu.world_pos()
                    sx, sy = shoulder.x, shoulder.y
                    # 将目标相对肩部的位移按增益放大，扩大两臂的运动范围
                    dx_rel, dy_rel = (tx - sx), (ty - sy)
                    if raise_this:
                        base_gain = ARM_RANGE_GAIN * ARM_RAISE_MODE_GAIN
                        txg = sx + dx_rel * base_gain
                        tyg = sy + dy_rel * base_gain
                        if tyg < sy - ELBOW_RAISE_MARGIN:
                            dir_gain = ARM_RANGE_GAIN * ARM_RAISE_EXTRA_GAIN * ARM_RAISE_MODE_GAIN
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
                        dx_off, dy_off = (txg - tx), (tyg - ty)
                        off = math.hypot(dx_off, dy_off)
                        limit = abs(rlen) * ROPE_STRETCH_RATIO_MAX
                        if off > limit and off > 1e-6:
                            k = limit / off
                            txg = tx + dx_off * k
                            tyg = ty + dy_off * k
                    else:
                        txg = tx
                        tyg = ty
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

            # 额外两个指尖索引
            INDEX_TIP = 8
            RING_TIP = 16
            index_tip_xy_screen = LM(hand.landmark[INDEX_TIP])
            ring_tip_xy_screen = LM(hand.landmark[RING_TIP])

            # 若尚未锁定，尝试建立初始映射：五指从左到右 → 目标顺序（考虑flip）
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
                    if not flip:
                        target_order = ['left_leg', 'left_hand', 'head_joint', 'right_hand', 'right_leg']
                    else:
                        target_order = ['right_leg', 'right_hand', 'head_joint', 'left_hand', 'left_leg']
                    locked_mapping = {fid: target_order[i] for i, fid in enumerate(sorted_ids)}
                    locked_wrist_screen = wrist_xy_screen
                    locked_active = True

            # 驱动腿：单段骨骼指向绳端
            def drive_leg_by_rope(side: str, anchor_xy_screen: tuple):
                if not anchor_xy_screen:
                    return
                ax, ay = anchor_xy_screen
                # 依据侧别选择校准后的绳长
                rlen = rope_len_map['left_leg'] if side == 'left' else rope_len_map['right_leg']
                rope_end_s = (ax, ay + rlen)
                tx, ty = screen_to_local(*rope_end_s)
                bn = 'left_leg' if side == 'left' else 'right_leg'
                leg = next(b for b in bones if b.name == bn)
                hip = leg.parent.world_pos() if leg.parent else Vec2(0,0)
                sx, sy = hip.x, hip.y
                # 目标角（全局）指向绳端
                vx, vy = tx - sx, ty - sy
                if not (abs(vx) < 1e-6 and abs(vy) < 1e-6):
                    glob = math.degrees(math.atan2(-vx, vy))
                    parent_angle = leg.parent.world_angle() if leg.parent else 0.0
                    bone_state[bn]['target'] = glob - parent_angle

            # 已锁定：根据固定映射驱动；若当前帧没有检测到任何手，则解除锁定
            if locked_active:
                tips_by_id = {
                    THUMB_TIP: thumb_tip_xy_screen,
                    INDEX_TIP: index_tip_xy_screen,
                    MIDDLE_TIP: middle_tip_xy_screen,
                    RING_TIP: ring_tip_xy_screen,
                    PINKY_TIP: pinky_tip_xy_screen,
                }
                # 若按下了校准请求：依据当前人偶初始姿态与手指位置，更新各部位绳长
                if calibrate_rope_request:
                    # 计算屏幕坐标的目标点（人偶端）
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
                    def world_to_screen(p):
                        return (origin_x + (-(p.x) if flip else p.x), origin_y + p.y)
                    wrist_left = wrist_screen(next(b for b in bones if b.name == 'left_lower_arm'))
                    wrist_right = wrist_screen(next(b for b in bones if b.name == 'right_lower_arm'))
                    foot_left = leg_tip_screen(next(b for b in bones if b.name == 'left_leg'))
                    foot_right = leg_tip_screen(next(b for b in bones if b.name == 'right_leg'))
                    j_head = next(b for b in bones if b.name == 'head').world_pos()
                    head_xy = world_to_screen(j_head)
                    # 汇总目标点
                    target_screen = {
                        'left_hand': wrist_left,
                        'right_hand': wrist_right,
                        'left_leg': foot_left,
                        'right_leg': foot_right,
                        'head_joint': head_xy,
                    }
                    # 根据锁定映射，计算各部位绳长（仅竖直方向差值）
                    for fid, to_name in locked_mapping.items():
                        ft = tips_by_id.get(fid)
                        tgt = target_screen.get(to_name)
                        if ft is None or tgt is None:
                            continue
                        rope_len_map[to_name] = max(-1000, min(1000, int(tgt[1] - ft[1])))
                    # 中指刚性连接到头关节：用中指与头关节的竖直距离校准 origin 偏移
                    if middle_tip_xy_screen is not None:
                        rope_len_head = int(head_xy[1] - middle_tip_xy_screen[1])
                        # j_head.y 为头关节在局部坐标下的 y；origin_y = middle_y + origin_offset_y
                        origin_offset_y = rope_len_head - int(j_head.y)
                    calibrate_rope_request = False
                if not neutral_frame_request:
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
                            # 头部仅用于连线；旋转由朝向控制
                            pass
                locked_wrist_screen = wrist_xy_screen

            # 手的朝向控制头部旋转（±10°）——用掌根横向线（食指MCP与小指MCP）
            INDEX_MCP = 5
            PINKY_MCP = 17
            idx_mcp = LM(hand.landmark[INDEX_MCP])
            pky_mcp = LM(hand.landmark[PINKY_MCP])
            dir_vx = idx_mcp[0] - pky_mcp[0]
            dir_vy = idx_mcp[1] - pky_mcp[1]
            orient_deg = math.degrees(math.atan2(dir_vy, dir_vx)) if not (dir_vx == 0 and dir_vy == 0) else 0.0
            if neutral_frame_request:
                # 本帧保持中立，不更新头/躯干角度
                pass
            else:
                # 头部 ±10° 跟随手的朝向
                head_delta = max(-10.0, min(10.0, orient_deg))
                b_head = next(b for b in bones if b.name == 'head')
                bone_state[b_head.name]['target'] = bone_state[b_head.name]['rest'] + head_delta
                # 躯干角度不直接跟随手的朝向；改为由绳子拉动与惯性产生倾斜
                b_body = next(b for b in bones if b.name == 'body')
                body_base = bone_state[b_body.name]['rest']
                # 1) 惯性倾斜：origin 加速度映射到角度
                # 计算当前帧 origin 速度与加速度（基于上一帧）
                ovx = origin_x - last_ox
                ovy = origin_y - last_oy
                ax = ovx - prev_ovx
                ay = ovy - prev_ovy
                inertial = max(-SWAY_LIMIT_DEG, min(SWAY_LIMIT_DEG, SWAY_ACC_X * ax + SWAY_ACC_Y * ay))
                body_delta = inertial
                hip_extra_tension = 0.0
                # 2) 绳子拉动：当腿被上拉到略微向上的角度时，继续作用到躯干上
                if locked_active and locked_mapping:
                    # 计算当前腿部末端屏幕 y 与绳端屏幕 y 的差
                    def leg_tip_screen(b_leg):
                        pos = b_leg.world_pos()
                        tip = b_leg.tip_offset()
                        wx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                        wy = origin_y + (pos.y + tip.y)
                        return int(wx), int(wy)
                    # 逆查找映射的指尖坐标
                    fid_left = next((fid for fid, name in locked_mapping.items() if name == 'left_leg'), None)
                    fid_right = next((fid for fid, name in locked_mapping.items() if name == 'right_leg'), None)
                    tips_by_id_local = {
                        4: thumb_tip_xy_screen,
                        8: index_tip_xy_screen,
                        12: middle_tip_xy_screen,
                        16: ring_tip_xy_screen,
                        20: pinky_tip_xy_screen,
                    }
                    # 左腿
                    try:
                        b_left_leg = next(b for b in bones if b.name == 'left_leg')
                        ly = leg_tip_screen(b_left_leg)[1]
                        if fid_left is not None:
                            ft = tips_by_id_local.get(fid_left)
                            if ft is not None:
                                rope_end_y = ft[1] + rope_len_map['left_leg']
                                upward_pull = max(0, ly - rope_end_y)  # 需要上拉的像素量
                                # 反转符号：左侧上拉→向右倾斜（正）
                                body_delta += BODY_TENSION_GAIN * upward_pull
                                # 髋部额外增益同步反转，使方向与躯干一致
                                hip_extra_tension += (HIP_TENSION_GAIN - BODY_TENSION_GAIN) * upward_pull
                    except StopIteration:
                        pass
                    # 右腿
                    try:
                        b_right_leg = next(b for b in bones if b.name == 'right_leg')
                        ry = leg_tip_screen(b_right_leg)[1]
                        if fid_right is not None:
                            ft = tips_by_id_local.get(fid_right)
                            if ft is not None:
                                rope_end_y = ft[1] + rope_len_map['right_leg']
                                upward_pull = max(0, ry - rope_end_y)
                                # 反转符号：右侧上拉→向左倾斜（负）
                                body_delta += -BODY_TENSION_GAIN * upward_pull
                                # 髋部额外增益同步反转，使方向与躯干一致
                                hip_extra_tension += -(HIP_TENSION_GAIN - BODY_TENSION_GAIN) * upward_pull
                    except StopIteration:
                        pass
                # 限幅并更新目标
                body_delta = max(-SWAY_LIMIT_DEG, min(SWAY_LIMIT_DEG, body_delta))
                bone_state[b_body.name]['target'] = body_base + body_delta
                # 二连杆耦合：腰部通过阻尼弹簧与躯干耦合，形成相位滞后与更明显的摆动
                b_waist = next(b for b in bones if b.name == 'waist')
                waist_base = bone_state[b_waist.name]['rest']
                # 使用当前帧的时间步长（回退到 30FPS）
                dt_frame = 1.0 / max(1.0, clock.get_fps() or 30.0)
                # 输入为躯干的倾斜目标（包含惯性与绳拉），腰部在输入与耦合项之间平衡
                K_in = 5.0   # 腰部接受输入的增益（增大以提升可见摆动）
                K_c = 3.2    # 与躯干目标之间的耦合刚度（略降低让腰部更自由）
                C_w = 2.0    # 腰部摆动的阻尼（降低以增强晃动幅度）
                # 加上竖直加速度对腰部的额外驱动，使得快速上下移动更显著摆动
                extra_drive = 0.5 * (SWAY_ACC_Y * ay)
                drive_target = body_delta + extra_drive + hip_extra_tension
                trunk_waist_vel += (K_in * drive_target - K_c * (trunk_waist_drive - body_delta) - C_w * trunk_waist_vel) * dt_frame
                trunk_waist_drive += trunk_waist_vel * dt_frame
                waist_delta = max(-SWAY_LIMIT_DEG * 1.6, min(SWAY_LIMIT_DEG * 1.6, trunk_waist_drive))
                bone_state[b_waist.name]['target'] = waist_base + waist_delta
            if middle_tip_xy_screen is not None:
                txo = middle_tip_xy_screen[0]
                tyo = middle_tip_xy_screen[1] + origin_offset_y
                dt_o = 1.0 / max(1.0, clock.get_fps() or 30.0)
                ax_o = ORIGIN_K * (txo - origin_x) - ORIGIN_C * origin_vx
                ay_o = ORIGIN_K * (tyo - origin_y) - ORIGIN_C * origin_vy + G
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
                ground_y = int(WIN_H * GROUND_Y_RATIO)
                try:
                    b_left_leg = next(b for b in bones if b.name == 'left_leg')
                    b_right_leg = next(b for b in bones if b.name == 'right_leg')
                    def tip_screen(b_leg):
                        pos = b_leg.world_pos()
                        tip = b_leg.tip_offset()
                        sx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                        sy = origin_y + (pos.y + tip.y)
                        return sx, sy
                    lx, ly = tip_screen(b_left_leg)
                    rx, ry = tip_screen(b_right_leg)
                    deepest = max(ly, ry)
                    if deepest >= ground_y:
                        origin_y -= (deepest - ground_y)
                        origin_vy = 0.0
                except StopIteration:
                    pass
                dx2 = origin_x - last_ox
                dy2 = origin_y - last_oy
                lim2 = ORIGIN_MAX_DELTA
                if dx2 > lim2:
                    origin_x = last_ox + lim2
                elif dx2 < -lim2:
                    origin_x = last_ox - lim2
                if dy2 > lim2:
                    origin_y = last_oy + lim2
                elif dy2 < -lim2:
                    origin_y = last_oy - lim2
                prev_ovx, prev_ovy = origin_x - last_ox, origin_y - last_oy
                last_ox, last_oy = origin_x, origin_y
        else:
            # 无手可用：解除链接锁定（人偶保持上一次位置），并开启下落与地面支撑
            locked_active = False
            locked_mapping = {}
            locked_wrist_screen = None
            # 下落（原点作为整体质心近似）
            dt_grav = 1.0 / max(1.0, clock.get_fps() or 30.0)
            origin_vy += G * dt_grav
            if origin_vy > ORIGIN_MAX_SPEED_Y:
                origin_vy = ORIGIN_MAX_SPEED_Y
            elif origin_vy < -ORIGIN_MAX_SPEED_Y:
                origin_vy = -ORIGIN_MAX_SPEED_Y
            ny = origin_y + origin_vy * dt_grav
            dy_grav = ny - last_oy
            lim = ORIGIN_MAX_DELTA
            if dy_grav > lim:
                ny = last_oy + lim
            elif dy_grav < -lim:
                ny = last_oy - lim
            origin_y = ny
            # 计算两脚在屏幕坐标系下的位置
            try:
                b_left_leg = next(b for b in bones if b.name == 'left_leg')
                b_right_leg = next(b for b in bones if b.name == 'right_leg')
                # tip = joint + local tip offset
                def tip_screen(b_leg):
                    pos = b_leg.world_pos()
                    tip = b_leg.tip_offset()
                    sx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                    sy = origin_y + (pos.y + tip.y)
                    return sx, sy
                lx, ly = tip_screen(b_left_leg)
                rx, ry = tip_screen(b_right_leg)
                ground_y = int(WIN_H * GROUND_Y_RATIO)
                # 地面支撑：若脚已压入地面，将原点上抬并清空竖直速度
                deepest = max(ly, ry)
                if deepest >= ground_y:
                    origin_y -= (deepest - ground_y)
                    origin_vy = 0.0
                    dy2 = origin_y - last_oy
                    lim2 = ORIGIN_MAX_DELTA
                    if dy2 > lim2:
                        origin_y = last_oy + lim2
                    elif dy2 < -lim2:
                        origin_y = last_oy - lim2
                    # 确定倒向方向：依据上一帧水平速度或左右脚的接触先后
                    if fall_side == 0:
                        # 使用上一帧水平位移近似方向；为简洁，此处随机选择
                        fall_side = -1 if (lx < rx) else 1
                    # 让躯干与髋部目标逐渐朝倒地姿态（±90°）
                    try:
                        b_body = next(b for b in bones if b.name == 'body')
                        b_waist = next(b for b in bones if b.name == 'waist')
                        base_body = bone_state[b_body.name]['rest']
                        base_waist = bone_state[b_waist.name]['rest']
                        target_fall = 90.0 * fall_side
                        # 倒地：目标直接朝 ±90° 偏转，阻尼弹簧会平滑逼近
                        bone_state[b_body.name]['target'] = base_body + target_fall
                        bone_state[b_waist.name]['target'] = base_waist + target_fall * 0.8
                    except StopIteration:
                        pass
            except StopIteration:
                pass
            prev_ovx, prev_ovy = origin_x - last_ox, origin_y - last_oy
            last_ox, last_oy = origin_x, origin_y

        # Physical integration for each bone
        dt_phys = 1.0 / max(1.0, clock.get_fps() or 30.0)
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
            # 对躯干角度进行显式限幅：围绕 rest ± SWAY_LIMIT_DEG
            if b.name == 'body':
                min_a = bone_state[b.name]['rest'] - SWAY_LIMIT_DEG
                max_a = bone_state[b.name]['rest'] + SWAY_LIMIT_DEG
                st['theta'] = max(min_a, min(max_a, st['theta']))
            b.angle_deg = st['theta']

        # 中立展示仅持续一帧
        neutral_frame_request = False

        # Draw background
        screen.fill(BG_COLOR)

        # Background: full camera frame (水平镜像)
        cam_surface = to_pygame_surface(frame)
        cam_surface = pygame.transform.flip(cam_surface, True, False)
        cam_surface = pygame.transform.smoothscale(cam_surface, (WIN_W, WIN_H))
        screen.blit(cam_surface, (0, 0))

        # Middle layer: semi-transparent shadow
        if shadow_surface:
            rect = shadow_surface.get_rect()
            rect.center = (WIN_W // 2, WIN_H // 2)
            screen.blit(shadow_surface, rect)

        # Ground: transparent
        ground_y = int(WIN_H * GROUND_Y_RATIO)
        try:
            gl = next(g for g in ghost_bones if g.name == 'left_leg')
            gr = next(g for g in ghost_bones if g.name == 'right_leg')
            ly = int(gl.world_pos().y + gl.tip_offset().y)
            ry = int(gr.world_pos().y + gr.tip_offset().y)
            deepest = max(ly, ry)
            ghost_origin[1] = ground_y - deepest
        except StopIteration:
            pass
        if tieshan_img:
            rect_ts = tieshan_img.get_rect()
            rect_ts.midbottom = (int(WIN_W * 0.25), ground_y)
            screen.blit(tieshan_img, rect_ts)

        # Right panel: puppet sprites (role 'niu'), support mirror draw if flip
        # 关闭蓝色端点标记；按 Space 切换骨骼叠加
        sm.draw(screen, origin_x, origin_y, show_bones=show_bones, flip=flip, show_endpoints=False)

        # Ghost pose: 绘制半透明虚影骨骼供对齐（不随 flip 翻转）
        ghost_color = (180, 160, 255)
        ghost_surface = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        for gb in ghost_bones:
            pos = gb.world_pos()
            tip = gb.tip_offset()
            jx = ghost_origin[0] + int(pos.x)
            jy = ghost_origin[1] + int(pos.y)
            tx = ghost_origin[0] + int(pos.x + tip.x)
            ty = ghost_origin[1] + int(pos.y + tip.y)
            pygame.draw.line(ghost_surface, (*ghost_color, 110), (jx, jy), (tx, ty), 3)
            pygame.draw.circle(ghost_surface, (*ghost_color, 110), (jx, jy), 5)
        screen.blit(ghost_surface, (0, 0))

        # Draw control strings from fingers to puppet joints if available
        if result and result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            def Lp(i):
                lm = hand.landmark[i]
                # 与摄像头层的水平镜像一致（X 取 WIN_W - x）
                return (WIN_W - int(lm.x * WIN_W), int(lm.y * WIN_H))
            # indices（五指指尖）
            THUMB_TIP = 4
            INDEX_TIP = 8
            MIDDLE_TIP = 12
            RING_TIP = 16
            PINKY_TIP = 20
            # wrists (lower-arm tips) in screen coordinates
            def wrist_screen(b_lower):
                pos = b_lower.world_pos()
                tip = b_lower.tip_offset()
                wx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                wy = origin_y + (pos.y + tip.y)
                return int(wx), int(wy)
            wrist_right = wrist_screen(next(b for b in bones if b.name == 'right_lower_arm'))
            wrist_left = wrist_screen(next(b for b in bones if b.name == 'left_lower_arm'))
            def leg_tip_screen(b_leg):
                pos = b_leg.world_pos()
                tip = b_leg.tip_offset()
                wx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                wy = origin_y + (pos.y + tip.y)
                return int(wx), int(wy)
            foot_left = leg_tip_screen(next(b for b in bones if b.name == 'left_leg'))
            foot_right = leg_tip_screen(next(b for b in bones if b.name == 'right_leg'))
            j_head = next(b for b in bones if b.name == 'head').world_pos()
            def world_to_screen(p):
                return (origin_x + (-(p.x) if flip else p.x), origin_y + p.y)
            def draw_string(ft, wrist_xy):
                # 直接画到手腕（绳子保持绷紧）
                pygame.draw.line(screen, (230,230,230), ft, wrist_xy, 2)
            # 若已锁定：按固定映射绘制五条绳子；否则按当前帧排序绘制参考
            head_xy = world_to_screen(j_head)
            def target_xy(name):
                if name == 'left_hand':
                    return wrist_left
                if name == 'right_hand':
                    return wrist_right
                if name == 'left_leg':
                    return foot_left
                if name == 'right_leg':
                    return foot_right
                if name == 'head_joint':
                    return head_xy
                return None
            if 'locked_mapping' in locals() and locked_mapping:
                for fid, to_name in locked_mapping.items():
                    ft = Lp(fid)
                    tx = target_xy(to_name)
                    if ft and tx:
                        draw_string(ft, tx)
            else:
                tips = [Lp(THUMB_TIP), Lp(INDEX_TIP), Lp(MIDDLE_TIP), Lp(RING_TIP), Lp(PINKY_TIP)]
                tips = [p for p in tips if p is not None]
                tips.sort(key=lambda p: p[0])
                targets = [foot_left, wrist_left, head_xy, wrist_right, foot_right]
                if flip:
                    targets = [foot_right, wrist_right, head_xy, wrist_left, foot_left]
                for i in range(min(len(tips), len(targets))):
                    draw_string(tips[i], targets[i])

        # Pose match：各部位按阈值归一化后计分，满足阈值则加分
        def puppet_match(bones, ghost_bones, ghost_origin, origin_x, origin_y, flip,
                         MATCH_THRESH_BODY, MATCH_THRESH_WAIST, MATCH_THRESH_OTHERS,
                         locked_active):
            def puppet_joint_tip_screen(b):
                pos = b.world_pos()
                tip = b.tip_offset()
                jx = origin_x + (-(pos.x) if flip else pos.x)
                jy = origin_y + pos.y
                tx = origin_x + (-(pos.x + tip.x) if flip else (pos.x + tip.x))
                ty = origin_y + (pos.y + tip.y)
                return (int(jx), int(jy)), (int(tx), int(ty))
            def ghost_joint_tip_screen(gb):
                pos = gb.world_pos()
                tip = gb.tip_offset()
                jx = ghost_origin[0] + pos.x
                jy = ghost_origin[1] + pos.y
                tx = ghost_origin[0] + pos.x + tip.x
                ty = ghost_origin[1] + pos.y + tip.y
                return (int(jx), int(jy)), (int(tx), int(ty))
            if not locked_active:
                return False
            total_n = 0.0
            count = 0
            gb_map = {gb.name: gb for gb in ghost_bones}
            for b in bones:
                gb = gb_map.get(b.name)
                if not gb:
                    continue
                (pj, pt) = puppet_joint_tip_screen(b)
                (gj, gt) = ghost_joint_tip_screen(gb)
                dx = pj[0] - gj[0]
                dy = pj[1] - gj[1]
                d2_j = dx*dx + dy*dy
                dx2 = pt[0] - gt[0]
                dy2 = pt[1] - gt[1]
                d2_t = dx2*dx2 + dy2*dy2
                thr = MATCH_THRESH_BODY if b.name == 'body' else (MATCH_THRESH_WAIST if b.name == 'waist' else MATCH_THRESH_OTHERS)
                total_n += (math.sqrt(d2_j) / thr) + (math.sqrt(d2_t) / thr)
                count += 2
            avg_n = (total_n / count) if count > 0 else 1e9
            return avg_n <= 1.0

        matched = puppet_match(bones, ghost_bones, ghost_origin, origin_x, origin_y, flip,
                                MATCH_THRESH_BODY, MATCH_THRESH_WAIST, MATCH_THRESH_OTHERS,
                                locked_active)
        if matched:
            try:
                import cv2
                vp = os.path.join(PROJECT_DIR, 'video', 'm1.mp4')
                cap_v = cv2.VideoCapture(vp)
                while True:
                    ok, fr = cap_v.read()
                    if not ok:
                        break
                    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    img = pygame.image.frombuffer(fr.tobytes(), (fr.shape[1], fr.shape[0]), 'RGB')
                    img = pygame.transform.smoothscale(img, (WIN_W, WIN_H))
                    screen.blit(img, (0, 0))
                    pygame.display.flip()
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT:
                            break
                cap_v.release()
            except Exception:
                pass
            running = False

        # HUD
        # 读取当前躯干、髋部角度用于 HUD 展示
        try:
            angle_body = next(b for b in bones if b.name == 'body').angle_deg
            angle_waist = next(b for b in bones if b.name == 'waist').angle_deg
        except StopIteration:
            angle_body = angle_waist = 0.0
        lines = [
            'Esc 退出 | R 重置 | 依赖: mediapipe/opencv',
            '画面: 摄像头底层 → 半透明Shadow → 人偶（Space 显示骨架）',
            '映射: 拇指→右手, 小指→左手, 中指→头部 | 快甩→切换贴图版本',
            f'贴图版本: {"flip" if anchors_is_flip else "非flip"} | 骨架: {"显示" if show_bones else "隐藏"}',
            f'角度: 躯干 {angle_body:.1f}° | 髋部 {angle_waist:.1f}°',
        ]
        y = 8
        for ln in lines:
            surf = font.render(ln, True, TEXT_COLOR)
            screen.blit(surf, (12, y))
            y += 22
        # 右上角显示当前得分
        score_txt = font.render(f'Score: {score}', True, (255, 235, 120))
        sw, sh = score_txt.get_size()
        screen.blit(score_txt, (WIN_W - sw - 12, 8))

        pygame.display.flip()

    cap.release()


if __name__ == '__main__':
    main()
