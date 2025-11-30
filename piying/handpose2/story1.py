import os
import time
import random
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

WIN_W, WIN_H = 1920, 1080
GROUND_Y_RATIO = 0.90

def _load(path):
    try:
        img = pygame.image.load(path)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in ('.png', '.tif', '.tiff'):
                img = img.convert_alpha()
            else:
                img = img.convert()
        except Exception:
            try:
                img = img.convert()
            except Exception:
                pass
        return img
    except Exception:
        return None

def _blit_midbottom(screen, surf, x, y, max_w=None, max_h=None):
    if surf is None:
        return
    img = surf
    if max_w and max_h:
        tw = max(1, min(max_w, img.get_width()))
        th = max(1, min(max_h, img.get_height()))
        img = pygame.transform.smoothscale(img, (tw, th))
    rect = img.get_rect()
    rect.midbottom = (x, y)
    screen.blit(img, rect)

def _draw_badges(screen, statuses):
    bad = _load(os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'bad.png'))
    great = _load(os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'Great.png'))
    y = 20
    pad = 20
    cur_x = WIN_W - 5
    imgs = []
    for i in range(5):
        imgs.append(great if (i < len(statuses) and statuses[i]) else bad)
    widths = [im.get_width() if im else 0 for im in imgs]
    total_w = sum(widths) + pad * 4
    start_x = WIN_W - total_w - 20 - 100
    x = start_x
    for i, im in enumerate(imgs):
        if im is None:
            continue
        rect = im.get_rect()
        rect.topleft = (x, y)
        screen.blit(im, rect)
        x += rect.width + pad

def _draw_subtitle(screen, surf):
    if surf is None:
        return
    h = int(WIN_H * 0.1)
    y0 = WIN_H - h
    img = pygame.transform.smoothscale(surf, (min(WIN_W, surf.get_width()), h))
    r = img.get_rect()
    r.centerx = WIN_W // 2
    r.bottom = WIN_H
    screen.blit(img, r)

def _load_gif_frames(path):
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return None
    try:
        im = Image.open(path)
        frames = []
        durs = []
        for f in ImageSequence.Iterator(im):
            dur = f.info.get('duration', 100) / 1000.0
            durs.append(max(0.01, float(dur)))
            mode = f.mode
            if mode != 'RGBA':
                f = f.convert('RGBA')
            data = f.tobytes()
            size = f.size
            surf = pygame.image.frombuffer(data, size, 'RGBA')
            frames.append(surf.copy())
        if not frames:
            return None
        return {'frames': frames, 'durations': durs}
    except Exception:
        return None

def _make_anim(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.gif':
        gf = _load_gif_frames(path)
        if gf:
            return {'is_anim': True, 'frames': gf['frames'], 'dur': gf['durations'], 'idx': 0, 'elapsed': 0.0}
    img = _load(path)
    return {'is_anim': False, 'static': img}

def _update_anim(anim, dt):
    if not anim or not anim.get('is_anim'):
        return
    anim['elapsed'] += dt
    while anim['elapsed'] >= anim['dur'][anim['idx']]:
        anim['elapsed'] -= anim['dur'][anim['idx']]
        anim['idx'] = (anim['idx'] + 1) % len(anim['frames'])

def _current_surface(anim):
    if not anim:
        return None
    if anim.get('is_anim'):
        return anim['frames'][anim['idx']]
    return anim.get('static')

def _run_dialog(screen, bg_path, left_path, right_path, zimu_dir, actions, badges):
    bg = _load(bg_path)
    if bg:
        bg = pygame.transform.smoothscale(bg, (WIN_W, WIN_H))
    left = _make_anim(left_path)
    right = _make_anim(right_path)
    zimu_paths = []
    dirp = os.path.join(PROJECT_DIR, 'vision_resources', 'instory', zimu_dir)
    try:
        names = sorted([n for n in os.listdir(dirp) if n.lower().endswith('.png')], key=lambda s: int(os.path.splitext(s)[0]))
        zimu_paths = [os.path.join(dirp, n) for n in names]
    except Exception:
        zimu_paths = []
    idx = 0
    start = time.time()
    ground_y = int(WIN_H * GROUND_Y_RATIO)
    runn = True
    last = time.time()
    while runn:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return False
        now = time.time()
        dt = now - last
        last = now
        if now - start >= 5.0:
            idx += 1
            start = now
        if idx >= len(zimu_paths):
            break
        if bg:
            screen.blit(bg, (0, 0))
        lp = actions.get((idx + 1, 'left'))
        rp = actions.get((idx + 1, 'right'))
        if lp is not None:
            left = _make_anim(lp)
        if rp is not None:
            right = _make_anim(rp)
        _update_anim(left, dt)
        _update_anim(right, dt)
        _blit_midbottom(screen, _current_surface(left), int(WIN_W * 0.25), ground_y)
        _blit_midbottom(screen, _current_surface(right), int(WIN_W * 0.75), ground_y)
        sub = _load(zimu_paths[idx])
        _draw_subtitle(screen, sub)
        _draw_badges(screen, badges)
        pygame.display.flip()
        pygame.time.delay(16)
    return True

def _show_title(screen, path, badges):
    img = _load(path)
    if img:
        img = pygame.transform.smoothscale(img, (WIN_W, WIN_H))
    t0 = time.time()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                return False
        if img:
            screen.blit(img, (0, 0))
        _draw_badges(screen, badges)
        pygame.display.flip()
        if time.time() - t0 >= 2.0:
            break
        pygame.time.delay(16)
    return True

def _run_match_external():
    try:
        import importlib.util
        mp = os.path.join(PROJECT_DIR, 'handpose2', 'match.py')
        spec = importlib.util.spec_from_file_location('handpose2_match', mp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        return True
    except Exception:
        return False

def _run_match2_external():
    try:
        import importlib.util
        mp = os.path.join(PROJECT_DIR, 'handpose2', 'match2.py')
        spec = importlib.util.spec_from_file_location('handpose2_match2', mp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        return True
    except Exception:
        return False

def _summary(screen):
    surf = pygame.Surface((WIN_W, WIN_H))
    surf.fill((20, 20, 22))
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    pygame.time.delay(1500)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    badges = [False, False, False, False, False]

    if not _show_title(screen, os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'title1.png'), badges):
        pygame.quit()
        return
    ok1 = _run_dialog(
        screen,
        os.path.join(PROJECT_DIR, 'vision_resources', 'inside_cave.png'),
        os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'maonv.png'),
        os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'sunwukong.png'),
        'zimu1',
        {
            (1, 'left'): os.path.join(PROJECT_DIR, 'video', 'maonv.gif'),
            (3, 'left'): os.path.join(PROJECT_DIR, 'video', 'maonv.gif'),
            (2, 'right'): os.path.join(PROJECT_DIR, 'video', 'sunwukong1.gif'),
        },
        badges
    )
    if not ok1:
        pygame.quit()
        return
    def _rand_tsgz():
        return os.path.join(PROJECT_DIR, 'video', random.choice(['tieshangongzhu1.gif','tieshangongzhu2.gif','tieshangongzhu3.gif']))
    def _rand_swk123():
        return os.path.join(PROJECT_DIR, 'video', random.choice(['sunwukong1.gif','sunwukong2.gif','sunwukong3.gif']))
    ok2 = _run_dialog(
        screen,
        os.path.join(PROJECT_DIR, 'vision_resources', 'inside_cave.png'),
        os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'tieshangongzhu.png'),
        os.path.join(PROJECT_DIR, 'vision_resources', 'instory', 'sunwukong.png'),
        'zimu2',
        {
            (1, 'left'): _rand_tsgz(),
            (3, 'left'): _rand_tsgz(),
            (5, 'left'): _rand_tsgz(),
            (8, 'left'): _rand_tsgz(),
            (2, 'right'): os.path.join(PROJECT_DIR, 'video', 'sunwukong4.gif'),
            (4, 'right'): _rand_swk123(),
            (6, 'right'): _rand_swk123(),
            (7, 'right'): _rand_swk123(),
        },
        badges
    )
    if not ok2:
        pygame.quit()
        return
    okm = _run_match_external()
    if okm:
        badges[0] = True
    _draw_badges(screen, badges)
    pygame.display.flip()

    for i in range(2, 6):
        if i == 2:
            _show_title(screen, os.path.join(PROJECT_DIR, 'vision_resources', 'instory', f'title{i}.png'), badges)
            dirp = os.path.join(PROJECT_DIR, 'vision_resources', 'instory', '2')
            try:
                names = sorted([
                    n for n in os.listdir(dirp)
                    if n.lower().endswith('.png')
                ], key=lambda s: int(os.path.splitext(s)[0]) if os.path.splitext(s)[0].isdigit() else s)
            except Exception:
                names = []
            t0 = time.time()
            idx = 0
            while idx < len(names):
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit(); return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        pygame.quit(); return
                p = os.path.join(dirp, names[idx])
                img = _load(p)
                if img:
                    img = pygame.transform.smoothscale(img, (WIN_W, WIN_H))
                    screen.blit(img, (0, 0))
                _draw_badges(screen, badges)
                pygame.display.flip()
                if time.time() - t0 >= 5.0:
                    idx += 1
                    t0 = time.time()
                pygame.time.delay(16)
            okm2 = _run_match2_external()
            if okm2:
                badges[1] = True
            _draw_badges(screen, badges)
            pygame.display.flip()
        else:
            _show_title(screen, os.path.join(PROJECT_DIR, 'vision_resources', 'instory', f'title{i}.png'), badges)

    _summary(screen)
    pygame.quit()

if __name__ == '__main__':
    main()
