import os
import sys
import json
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

WIN_W, WIN_H = 1280, 720
BG = (18, 18, 20)
FG = (240, 240, 240)
TARGET_INIT_HEIGHT = 330
SCALE_MIN = 0.1
SCALE_MAX = 5.0
SCALE_STEP = 0.05

def json_path_for(subpath):
    d, f = os.path.split(subpath)
    n, _ = os.path.splitext(f)
    jp = os.path.join(PROJECT_DIR, 'collision_volume', d, n + '.json')
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    return jp

def load_image_and_shapes(subpath):
    ip = os.path.join(PROJECT_DIR, 'vision_resources', subpath)
    img = None
    try:
        img = pygame.image.load(ip).convert_alpha()
    except Exception:
        img = None
    shapes = []
    scale = 1.0
    jp = json_path_for(subpath)
    if os.path.exists(jp):
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ss = data.get('shapes') or []
            for s in ss:
                x = float(s.get('x', 0.0))
                y = float(s.get('y', 0.0))
                w = float(s.get('w', 0.0))
                h = float(s.get('h', 0.0))
                r = int(s.get('r', 12))
                shapes.append({'x': x, 'y': y, 'w': w, 'h': h, 'r': r})
            scale = float(data.get('scale', 1.0))
        except Exception:
            shapes = []
            scale = 1.0
    if img and not shapes:
        iw, ih = img.get_size()
        shapes.append({'x': 0.0, 'y': 0.0, 'w': float(iw), 'h': float(ih), 'r': 12})
    if img and (not (scale > 0.0)):
        ih = img.get_height()
        if ih > 0:
            scale = max(SCALE_MIN, min(SCALE_MAX, TARGET_INIT_HEIGHT / ih))
    return img, shapes, scale

def save_shapes(subpath, shapes, scale):
    jp = json_path_for(subpath)
    data = {'image': subpath, 'scale': float(scale), 'shapes': shapes}
    try:
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def draw_rounded(surface, rect, color, radius):
    pygame.draw.rect(surface, color, rect, border_radius=max(0, int(radius)))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Collision Volume Demo')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('consolas', 18)

    input_txt = 'test1.TIF'
    cur_img_path = input_txt
    img, shapes, img_scale = load_image_and_shapes(cur_img_path)
    first_run = True
    if img and first_run:
        ih = img.get_height()
        if ih > 0:
            img_scale = max(SCALE_MIN, min(SCALE_MAX, TARGET_INIT_HEIGHT / ih))
        first_run = False
    sel = 0

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    cur_img_path = input_txt.strip()
                    img, shapes, img_scale = load_image_and_shapes(cur_img_path)
                    sel = 0
                elif event.key == pygame.K_BACKSPACE:
                    if len(input_txt) > 0:
                        input_txt = input_txt[:-1]
                elif event.key == pygame.K_TAB:
                    if shapes:
                        sel = (sel + 1) % len(shapes)
                elif event.key == pygame.K_a:
                    if img:
                        iw, ih = img.get_size()
                        shapes.append({'x': 0.0, 'y': 0.0, 'w': float(iw), 'h': float(ih), 'r': 12})
                        sel = len(shapes) - 1
                        save_shapes(cur_img_path, shapes, img_scale)
                elif event.key == pygame.K_z or event.key == pygame.K_x:
                    pass
                else:
                    ch = event.unicode
                    if ch:
                        if ch.isalnum() or ch in '/._-':
                            input_txt += ch

        screen.fill(BG)
        cx, cy = WIN_W // 2, WIN_H // 2
        ix, iy = cx, cy
        top_left = (0, 0)
        if img:
            iw, ih = img.get_size()
            tw, th = max(1, int(iw * img_scale)), max(1, int(ih * img_scale))
            disp = pygame.transform.smoothscale(img, (tw, th))
            ir = disp.get_rect()
            ir.center = (cx, cy)
            ix, iy = ir.topleft
            screen.blit(disp, ir)

        if img and shapes:
            keys = pygame.key.get_pressed()
            step_xy = 2.0
            step_size = 4.0
            step_r = 2
            s = shapes[sel]
            if keys[pygame.K_LEFT]:
                s['x'] = s['x'] - step_xy
            if keys[pygame.K_RIGHT]:
                s['x'] = s['x'] + step_xy
            if keys[pygame.K_UP]:
                s['y'] = s['y'] - step_xy
            if keys[pygame.K_DOWN]:
                s['y'] = s['y'] + step_xy
            # 缩放参照点为形状中心
            cx = s['x'] + s['w'] * 0.5
            cy = s['y'] + s['h'] * 0.5
            # 横向缩放：[ 缩小, ] 放大
            if keys[pygame.K_LEFTBRACKET]:
                new_w = max(1.0, s['w'] - step_size)
                s['x'] = cx - new_w * 0.5
                s['w'] = new_w
            if keys[pygame.K_RIGHTBRACKET]:
                new_w = s['w'] + step_size
                s['x'] = cx - new_w * 0.5
                s['w'] = new_w
            # 纵向缩放：; 缩小, ' 放大
            if keys[pygame.K_SEMICOLON]:
                new_h = max(1.0, s['h'] - step_size)
                s['y'] = cy - new_h * 0.5
                s['h'] = new_h
            if keys[pygame.K_QUOTE]:
                new_h = s['h'] + step_size
                s['y'] = cy - new_h * 0.5
                s['h'] = new_h
            if keys[pygame.K_q]:
                s['r'] = max(0, int(s['r']) - step_r)
            if keys[pygame.K_e]:
                s['r'] = int(s['r']) + step_r
            if keys[pygame.K_z]:
                img_scale = max(SCALE_MIN, img_scale - SCALE_STEP)
            if keys[pygame.K_x]:
                img_scale = min(SCALE_MAX, img_scale + SCALE_STEP)
            save_shapes(cur_img_path, shapes, img_scale)

            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            for i, sh in enumerate(shapes):
                rx = int(ix + sh['x'] * img_scale)
                ry = int(iy + sh['y'] * img_scale)
                rw = int(sh['w'] * img_scale)
                rh = int(sh['h'] * img_scale)
                color = (255, 80, 80, 140) if i == sel else (255, 80, 80, 90)
                draw_rounded(overlay, pygame.Rect(rx, ry, rw, rh), color, sh['r'])
            screen.blit(overlay, (0, 0))

        hud = [
            '输入 vision_resources 子路径并回车切换图片，如 niu/head.png',
            f'当前: {cur_img_path}  缩放: {img_scale:.2f}  形状数: {len(shapes) if shapes else 0}  当前索引: {sel if shapes else -1}',
            '位置: ← → ↑ ↓  横向缩放: [ ]  纵向缩放: ; \'  圆角: Q/E  新增形状: A  切换: Tab  图片缩放: Z/X'
        ]
        y = 10
        for ln in hud:
            t = font.render(ln, True, FG)
            screen.blit(t, (12, y))
            y += 22
        inp = font.render('路径: ' + input_txt, True, (255, 235, 120))
        screen.blit(inp, (12, y))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()