import os
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

WIN_W, WIN_H = 1920, 1080
BG_COLOR = (18, 18, 20)
TEXT_COLOR = (240, 240, 240)

TARGET_RECT = pygame.Rect(1332, 333, 300, 414)

def to_surface(frame_bgr):
    import numpy as np
    frame_rgb = frame_bgr[:, :, ::-1]
    h, w = frame_rgb.shape[:2]
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')

def _safe_load(path):
    try:
        img = pygame.image.load(path)
        try:
            img = img.convert_alpha()
        except Exception:
            try:
                img = img.convert()
            except Exception:
                pass
        return img
    except Exception:
        return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    try:
        import cv2
        import mediapipe as mp
    except Exception as e:
        font = pygame.font.SysFont('consolas', 18)
        msg = [
            '缺少依赖：请先安装 handpose2/requirements.txt',
            'pip install -r handpose2/requirements.txt',
            f'ImportError: {e}',
        ]
        running = True
        while running:
            screen.fill(BG_COLOR)
            y = 40
            for t in msg:
                surf = font.render(t, True, TEXT_COLOR)
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
    hands = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 背景：2bg 80% 透明度
    bg = _safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'instory', '2bg.png'))
    if bg:
        bg = pygame.transform.smoothscale(bg, (WIN_W, WIN_H))
        bg.set_alpha(204)

    # 受控图片：xie.png（初始在左半屏竖直中心）
    xie = _safe_load(os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'xie.png'))
    xie_pos = [int(WIN_W * 0.25), WIN_H // 2]
    xie_angle = 0.0

    def palm_angle_deg(hand):
        try:
            idx = hand.landmark[5]
            pky = hand.landmark[17]
            vx = (idx.x - pky.x)
            vy = (idx.y - pky.y)
            if vx == 0 and vy == 0:
                return 0.0
            # 与摄像头层水平镜像一致（X 取 WIN_W - x），角度同 match 逻辑
            import math
            return math.degrees(math.atan2(vy, vx))
        except Exception:
            return 0.0

    def mostly_inside(rect_img):
        inter = rect_img.clip(TARGET_RECT)
        if rect_img.width * rect_img.height <= 0:
            return False
        ratio = (inter.width * inter.height) / (rect_img.width * rect_img.height)
        return ratio >= 0.6

    success = False
    running = True
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

        cam_surface = to_surface(frame)
        cam_surface = pygame.transform.flip(cam_surface, True, False)
        cam_surface = pygame.transform.smoothscale(cam_surface, (WIN_W, WIN_H))
        screen.blit(cam_surface, (0, 0))
        if bg:
            screen.blit(bg, (0, 0))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        hand = result.multi_hand_landmarks[0] if (result and result.multi_hand_landmarks) else None
        if hand is not None:
            h_tip = hand.landmark[12]
            fx = WIN_W - int(h_tip.x * WIN_W)
            fy = int(h_tip.y * WIN_H)
            xie_pos[0] = fx
            xie_pos[1] = fy
            xie_angle = palm_angle_deg(hand)

        if xie is not None:
            rot = pygame.transform.rotozoom(xie, -xie_angle, 1.0)
            rect = rot.get_rect()
            rect.center = (xie_pos[0], xie_pos[1])
            screen.blit(rot, rect)
            if mostly_inside(rect):
                success = True

        # 目标区域边框隐藏（保留检测逻辑，不绘制）

        pygame.display.flip()

        if success:
            try:
                vp = os.path.join(PROJECT_DIR, 'video', 'yun1.mp4')
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

    cap.release()

if __name__ == '__main__':
    main()
