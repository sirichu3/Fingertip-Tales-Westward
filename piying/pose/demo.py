import os
import sys
import json
import time
import math
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# import bone geom puppet
sys.path.append(os.path.join(PROJECT_DIR, 'bone'))
from geom_puppet import GeomPuppet
from sprite_attach import SpriteManager


WIN_W, WIN_H = 960, 720
BG_COLOR = (18, 18, 20)
BONE_COLOR = (0, 200, 255)
SEL_COLOR = (255, 180, 0)


def dist2(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def desired_global_angle_deg(pivot_x, pivot_y, mouse_x, mouse_y):
    vx = mouse_x - pivot_x
    vy = mouse_y - pivot_y
    if vx == 0 and vy == 0:
        return None
    # Bone local axis is +Y; world angle s.t. rotated (0,1) points to (vx,vy)
    return math.degrees(math.atan2(-vx, vy))


def save_pose(path_dir, origin, bones):
    angles = {b.name: float(b.angle_deg) for b in bones}
    ts = time.strftime('%Y%m%d_%H%M%S')
    data = {
        'origin': [int(origin[0]), int(origin[1])],
        'angles': angles,
        'timestamp': ts,
    }
    os.makedirs(path_dir, exist_ok=True)
    out_path = os.path.join(path_dir, f'pose_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption('Pose Demo — 鼠标拖动骨骼并保存姿态')
    clock = pygame.time.Clock()

    puppet = GeomPuppet()
    bones = puppet.bones()
    # 记录左右上臂索引，便于在拖动时根据鼠标左右侧动态切换控制侧
    name_to_index = {b.name: i for i, b in enumerate(bones)}
    idx_left_upper = name_to_index.get('left_upper_arm')
    idx_right_upper = name_to_index.get('right_upper_arm')

    # 角色贴图渲染控制（按数字查看渲染情况）
    ROLES = ['niu', 'sun', 'tu']
    role_index = 0
    role = ROLES[role_index]
    anchors_path = os.path.join(PROJECT_DIR, 'bone', f'anchors_{role}.json')
    sm = SpriteManager(puppet, anchors_path)
    show_sprites = False

    origin = [WIN_W // 2, WIN_H // 2 - 60]
    selected_idx = 0
    dragging = False

    font = pygame.font.SysFont('consolas', 18)

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    selected_idx = (selected_idx + 1) % len(bones)
                elif event.key == pygame.K_r:
                    puppet.reset_pose()
                elif event.key == pygame.K_s:
                    out = save_pose(BASE_DIR, origin, bones)
                    print(f'Saved pose: {out}')
                # 数字键切换角色渲染：1=niu, 2=sun, 3=tu；0=关闭渲染
                elif event.key == pygame.K_1:
                    role_index = 0
                    role = ROLES[role_index]
                    anchors_path = os.path.join(PROJECT_DIR, 'bone', f'anchors_{role}.json')
                    sm = SpriteManager(puppet, anchors_path)
                    show_sprites = True
                elif event.key == pygame.K_2:
                    role_index = 1
                    role = ROLES[role_index]
                    anchors_path = os.path.join(PROJECT_DIR, 'bone', f'anchors_{role}.json')
                    sm = SpriteManager(puppet, anchors_path)
                    show_sprites = True
                elif event.key == pygame.K_3:
                    role_index = 2
                    role = ROLES[role_index]
                    anchors_path = os.path.join(PROJECT_DIR, 'bone', f'anchors_{role}.json')
                    sm = SpriteManager(puppet, anchors_path)
                    show_sprites = True
                elif event.key == pygame.K_0:
                    show_sprites = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # collect candidates within threshold
                candidates = []
                thresh2 = 14 * 14
                for i, b in enumerate(bones):
                    pos = b.world_pos()
                    jx, jy = origin[0] + int(pos.x), origin[1] + int(pos.y)
                    d2 = dist2(mx, my, jx, jy)
                    if d2 <= thresh2:
                        candidates.append((i, jx, jy, d2, b.name))
                if candidates:
                    # choose closest distance first
                    min_d2 = min(c[3] for c in candidates)
                    close = [c for c in candidates if abs(c[3] - min_d2) <= 1e-6]
                    if len(close) == 1:
                        selected_idx = close[0][0]
                    else:
                        # disambiguate overlapping joints by mouse side: left_* vs right_*
                        # assume all close share same pivot coords
                        jx = close[0][1]
                        lefts = [c for c in close if c[4].startswith('left_')]
                        rights = [c for c in close if c[4].startswith('right_')]
                        if lefts and rights:
                            if mx < jx:
                                selected_idx = lefts[0][0]
                            else:
                                selected_idx = rights[0][0]
                        else:
                            # fallback to the first
                            selected_idx = close[0][0]
                    dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = event.pos
                # 若当前选中为左右上臂之一，则根据鼠标在肩关节左右侧动态切换控制对象
                if selected_idx in (idx_left_upper, idx_right_upper):
                    if idx_left_upper is not None and idx_right_upper is not None:
                        # 两侧的肩关节在几何上重叠，取任一的关节位置作为判定基准
                        posL = bones[idx_left_upper].world_pos()
                        jx = origin[0] + int(posL.x)
                        # 鼠标在关节左侧→控制左臂；右侧→控制右臂
                        selected_idx = idx_left_upper if mx < jx else idx_right_upper
                b = bones[selected_idx]
                pos = b.world_pos()
                jx, jy = origin[0] + int(pos.x), origin[1] + int(pos.y)
                deg = desired_global_angle_deg(jx, jy, mx, my)
                if deg is not None:
                    parent_angle = b.parent.world_angle() if b.parent else 0.0
                    b.angle_deg = deg - parent_angle

        # draw
        screen.fill(BG_COLOR)

        # 先渲染贴图，再覆盖骨骼线条，便于观察当前姿态下的角色效果
        if show_sprites:
            sm.draw(screen, origin[0], origin[1], show_bones=False)

        for i, b in enumerate(bones):
            pos = b.world_pos()
            jx, jy = origin[0] + int(pos.x), origin[1] + int(pos.y)
            tip = b.tip_offset()
            tx, ty = origin[0] + int(pos.x + tip.x), origin[1] + int(pos.y + tip.y)
            col = SEL_COLOR if i == selected_idx else BONE_COLOR
            pygame.draw.line(screen, col, (jx, jy), (tx, ty), 3)
            pygame.draw.circle(screen, col, (jx, jy), 5)

        # HUD
        lines = [
            f'选中: {bones[selected_idx].name}  角色渲染: {"开启" if show_sprites else "关闭"} ({role if show_sprites else "-"})',
            '左键选关节并拖动旋转 | Tab 切换 | R 重置 | S 保存姿态',
            '数字键查看角色渲染: 1=niu 2=sun 3=tu | 0 关闭渲染',
        ]
        y = 8
        for ln in lines:
            surf = font.render(ln, True, (240, 240, 240))
            screen.blit(surf, (8, y))
            y += 22

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()