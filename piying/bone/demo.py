import os
import sys
import json
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Allow importing sibling modules if needed
sys.path.append(BASE_DIR)

from geom_puppet import GeomPuppet
from sprite_attach import SpriteManager


ROLES = ['niu', 'niu_flip', 'sun', 'sun_flip', 'tu', 'tu_flip']


def ensure_role_anchors(role: str):
    anchors_path = os.path.join(BASE_DIR, f"anchors_{role}.json")
    if os.path.exists(anchors_path):
        return anchors_path
    # Create initial anchors mapping to vision_resources/<role>
    parts = [
        'head', 'body', 'waist',
        'left_upper_arm', 'left_lower_arm',
        'right_upper_arm', 'right_lower_arm',
        'left_leg', 'right_leg',
    ]
    anchors = {}
    for p in parts:
        # map to ../vision_resources/<role>/<file>.png (relative to bone folder)
        img_name = None
        # files present in vision_resources follow these names
        if p == 'body':
            img_name = 'body.png'
        elif p == 'head':
            img_name = 'head.png'
        elif p == 'waist':
            img_name = 'waist.png'
        elif p.endswith('upper_arm'):
            img_name = 'leftarm.png' if p.startswith('left_') else 'rightarm.png'
        elif p.endswith('lower_arm'):
            img_name = 'lefthand.png' if p.startswith('left_') else 'righthand.png'
        elif p.endswith('leg'):
            img_name = 'leftfoot.png' if p.startswith('left_') else 'rightfoot.png'

        rel_img = None
        if img_name:
            rel_img = os.path.join('..', 'vision_resources', role, img_name)
        anchors[p] = {"image": rel_img, "dx": 0.0, "dy": 0.0, "rot": 0.0, "scale": 1.0}

    data = {"__config__": {"autoscale": True}, "anchors": anchors}
    with open(anchors_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return anchors_path


def main():
    pygame.init()
    W, H = 900, 700
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Bone Demo - Multi Role Binding (1:niu 2:niu_flip 3:sun 4:sun_flip 5:tu 6:tu_flip)')
    clock = pygame.time.Clock()

    role_index = 0
    role = ROLES[role_index]

    def make_session(current_role):
        puppet = GeomPuppet()
        anchors_path = ensure_role_anchors(current_role)
        sm = SpriteManager(puppet, anchors_path)
        return puppet, sm

    puppet, sm = make_session(role)
    bones = puppet.bones()
    selected = 0
    show_bones = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sm.save_anchors()
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sm.save_anchors()
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_bones = not show_bones
                elif event.key == pygame.K_TAB:
                    selected = (selected + 1) % len(bones)
                elif event.key == pygame.K_s:
                    sm.save_anchors()
                elif event.key == pygame.K_1:
                    role_index = 0
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0
                elif event.key == pygame.K_2:
                    role_index = 1
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0
                elif event.key == pygame.K_3:
                    role_index = 2
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0
                elif event.key == pygame.K_4:
                    role_index = 3
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0
                elif event.key == pygame.K_5:
                    role_index = 4
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0
                elif event.key == pygame.K_6:
                    role_index = 5
                    role = ROLES[role_index]
                    puppet, sm = make_session(role)
                    bones = puppet.bones()
                    selected = 0

        keys = pygame.key.get_pressed()
        bname = bones[selected].name
        step_xy = 2.0
        step_rot = 1.0
        step_scale = 0.05

        if keys[pygame.K_LEFT]:
            sm.adjust(bname, 'dx', -step_xy)
        if keys[pygame.K_RIGHT]:
            sm.adjust(bname, 'dx', step_xy)
        if keys[pygame.K_UP]:
            sm.adjust(bname, 'dy', -step_xy)
        if keys[pygame.K_DOWN]:
            sm.adjust(bname, 'dy', step_xy)
        if keys[pygame.K_q]:
            sm.adjust(bname, 'rot', -step_rot)
        if keys[pygame.K_e]:
            sm.adjust(bname, 'rot', step_rot)
        if keys[pygame.K_z]:
            sm.adjust(bname, 'scale', -step_scale)
        if keys[pygame.K_x]:
            sm.adjust(bname, 'scale', step_scale)
        # 对齐：将当前部件的自定义 pivot 挪到关节上（dx=dy=0）
        if keys[pygame.K_j]:
            sm.snap_pivot_to_joint(bname)
        # 对齐所有：将所有贴图的 pivot 挪到各自骨骼关节上
        if keys[pygame.K_a]:
            sm.snap_all_pivots_to_joint()

        screen.fill((18, 18, 18))
        cx, cy = W // 2, H // 2 - 60
        sm.draw(screen, cx, cy, show_bones=show_bones)

        # UI overlay (top-left): add rotate/scale help
        font = pygame.font.SysFont('consolas', 18)
        hud_lines = [
            f'角色: {role}  部件: {bname}  [1/2/3/4/5/6 切换角色, Tab 下一个]',
            '旋转: Q/E    缩放: Z/X    移动: ↑ ↓ ← →    骨架: Space    保存: S',
            '对齐当前部件 pivot→关节: J    对齐所有: A',
        ]
        y = 14
        for line in hud_lines:
            txt = font.render(line, True, (240, 240, 240))
            screen.blit(txt, (14, y))
            y += 22

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
