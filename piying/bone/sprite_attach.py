import json
import os
import math
import pygame


class SpriteManager:
    def __init__(self, puppet, anchors_path: str):
        self.puppet = puppet
        self.anchors_path = anchors_path
        self.anchors = {}
        self.images = {}
        self.__config__ = {"autoscale": True}
        self.load_anchors()
        self.load_images()
        self.apply_initial_autoscale_if_needed()

    def load_anchors(self):
        if os.path.exists(self.anchors_path):
            with open(self.anchors_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.__config__ = data.get("__config__", {"autoscale": True})
            self.anchors = data.get("anchors", {})
        else:
            self.anchors = {}

    def save_anchors(self):
        data = {"__config__": self.__config__, "anchors": self.anchors}
        with open(self.anchors_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._sync_counterpart_scales()

    def disable_autoscale(self):
        if self.__config__.get("autoscale", True):
            self.__config__["autoscale"] = False
            self.save_anchors()

    def load_images(self):
        base_dir = os.path.dirname(self.anchors_path)
        for b in self.puppet.bones():
            a = self.anchors.get(b.name)
            if not a:
                continue
            path = a.get("image")
            if not path:
                continue
            abs_path = os.path.normpath(os.path.join(base_dir, path))
            if os.path.exists(abs_path):
                img = pygame.image.load(abs_path).convert_alpha()
                self.images[b.name] = img

    def get_anchor(self, bone_name: str):
        a = self.anchors.get(bone_name)
        if not a:
            a = {"image": None, "dx": 0.0, "dy": 0.0, "rot": 0.0, "scale": 1.0, "use_joint_pivot": True}
            self.anchors[bone_name] = a
        return a

    def apply_initial_autoscale_if_needed(self):
        if not self.__config__.get("autoscale", True):
            return
        # Simple heuristic: scale textures to bone width/length proportions
        for b in self.puppet.bones():
            img = self.images.get(b.name)
            if not img:
                continue
            a = self.get_anchor(b.name)
            if a.get("scale", 1.0) != 1.0:
                continue
            iw, ih = img.get_width(), img.get_height()
            # Fit height to bone length, keep aspect
            if ih > 0:
                scale = b.length / ih
                a["scale"] = scale
        # do not save here; will persist when user adjusts or presses S

    def draw(self, surface, origin_x: int, origin_y: int, show_bones: bool = False, flip: bool = False, show_endpoints: bool = False):
        name_to_bone = {b.name: b for b in self.puppet.bones()}
        if not flip:
            order_top = [
                'right_lower_arm',
                'right_upper_arm',
                'waist',
                'right_leg',
                'left_leg',
                'body',
                'left_lower_arm',
                'left_upper_arm',
                'head',
            ]
        else:
            order_top = [
                'left_lower_arm',
                'left_upper_arm',
                'waist',
                'left_leg',
                'right_leg',
                'body',
                'right_lower_arm',
                'right_upper_arm',
                'head',
            ]
        draw_order = [name_to_bone[n] for n in order_top if n in name_to_bone]
        for b in reversed(draw_order):
            pos = b.world_pos()
            x = float(origin_x) + (-pos.x if flip else pos.x)
            y = float(origin_y) + pos.y
            angle = float(b.world_angle())
            img = self.images.get(b.name)
            if not img:
                continue

            a = self.get_anchor(b.name)
            raw_scale = float(a.get("scale", 1.0))
            if not (raw_scale > 0):
                raw_scale = 1.0
            iw0, ih0 = img.get_size()
            max_safe_scale = 4096.0 / max(1.0, max(iw0, ih0))
            scale = max(0.05, min(raw_scale, max_safe_scale))

            rot_anchor = float(a.get("rot", 0.0))
            dx, dy = float(a.get("dx", 0.0)), float(a.get("dy", 0.0))
            flip_h_anchor = bool(a.get("flip_h", False))

            # Bone local offset rotated
            rad = math.radians(180.0 + angle)
            dxr = dx * math.cos(rad) - dy * math.sin(rad)
            dyr = dx * math.sin(rad) + dy * math.cos(rad)
            dx_eff = -dxr if flip else dxr
            dy_eff = dyr

            # Pivot
            use_joint_pivot = a.get("use_joint_pivot", True)
            pivot_rx = float(self._default_pivot_rx(b.name)) if use_joint_pivot else float(a.get("pivot_rx", 0.5))
            pivot_ry = float(self._default_pivot_ry(b.name)) if use_joint_pivot else float(a.get("pivot_ry", 0.5))

            # Scale & flip
            iw, ih = img.get_size()
            tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
            pre = pygame.transform.smoothscale(img, (tw, th))
            effective_flip = bool(flip) ^ bool(flip_h_anchor)
            if effective_flip:
                pre = pygame.transform.flip(pre, True, False)

            # Rotate
            rot_total = rot_anchor - angle
            rotated = pygame.transform.rotate(pre, rot_total)

            # Pivot offset in pre-scaled image
            pw, ph = pre.get_size()
            px = pw * (pivot_rx if not effective_flip else (1.0 - pivot_rx))
            py = ph * pivot_ry
            pvx = px - pw / 2.0
            pvy = py - ph / 2.0

            prad = math.radians(rot_total)
            rvx = pvx * math.cos(prad) - pvy * math.sin(prad)
            rvy = pvx * math.sin(prad) + pvy * math.cos(prad)

            # World center (bone pos + local offset)
            center_x = x + dx_eff
            center_y = y + dy_eff

            # Final rotated image center
            rotated_cx = center_x + rvx
            rotated_cy = center_y - rvy
            blit_x = int(round(rotated_cx))
            blit_y = int(round(rotated_cy))
            rx, ry = rotated.get_rect(center=(blit_x, blit_y)).topleft
            surface.blit(rotated, (rx, ry))

            # Debug: pivot and tip
            if show_endpoints:
                # Pivot world pos
                pivot_x = rotated_cx + rvx
                pivot_y = rotated_cy + rvy

                # Tip
                tip_rx = float(a.get("tip_rx", self._default_tip_rx(b.name)))
                tip_ry = float(a.get("tip_ry", self._default_tip_ry(b.name)))
                tx = pw * (tip_rx if not effective_flip else (1.0 - tip_rx))
                ty = ph * tip_ry
                tvx = tx - pw / 2.0
                tvy = ty - ph / 2.0
                tvx_r = tvx * math.cos(prad) - tvy * math.sin(prad)
                tvy_r = tvx * math.sin(prad) + tvy * math.cos(prad)
                tip_x = rotated_cx + tvx_r
                tip_y = rotated_cy + tvy_r

                pygame.draw.circle(surface, (0, 120, 255), (int(round(pivot_x)), int(round(pivot_y))), 5)
                pygame.draw.circle(surface, (0, 120, 255), (int(round(tip_x)), int(round(tip_y))), 5)

        # Skeleton overlay
        if show_bones:
            for b in self.puppet.bones():
                pos = b.world_pos()
                x = int(round(float(origin_x) + (-pos.x if flip else pos.x)))
                y = int(round(float(origin_y) + pos.y))
                pygame.draw.circle(surface, (0, 200, 255), (x, y), 4)
                tip = b.tip_offset()
                tx = int(round(float(origin_x) + (-(pos.x + tip.x) if flip else (pos.x + tip.x))))
                ty = int(round(float(origin_y) + pos.y + tip.y))
                pygame.draw.line(surface, (0, 200, 255), (x, y), (tx, ty), 2)

    def _default_pivot_rx(self, bone_name: str) -> float:
        # Heuristics: arms/legs/waist pivot at top center, body center
        if bone_name in ('left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg', 'waist', 'head'):
            return 0.5
        if bone_name == 'body':
            return 0.5
        return 0.5

    def _default_pivot_ry(self, bone_name: str) -> float:
        if bone_name in ('left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg', 'waist'):
            return 0.0
        if bone_name in ('body',):
            return 0.5
        if bone_name in ('head',):
            # Neck usually near top of head sprite
            return 0.1
        return 0.5

    def _default_tip_rx(self, bone_name: str) -> float:
        # Tip assumed at bottom center for limbs; body center; head bottom
        if bone_name in ('left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg'):
            return 0.5
        if bone_name in ('waist', 'body'):
            return 0.5
        if bone_name in ('head',):
            return 0.5
        return 0.5

    def _default_tip_ry(self, bone_name: str) -> float:
        if bone_name in ('left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm', 'left_leg', 'right_leg'):
            return 1.0
        if bone_name in ('waist',):
            return 1.0
        if bone_name in ('body',):
            return 0.5
        if bone_name in ('head',):
            return 1.0
        return 0.5

    def adjust(self, bone_name: str, kind: str, amount: float):
        a = self.get_anchor(bone_name)
        if kind == 'dx':
            a['dx'] = float(a.get('dx', 0.0)) + amount
        elif kind == 'dy':
            a['dy'] = float(a.get('dy', 0.0)) + amount
        elif kind == 'rot':
            a['rot'] = float(a.get('rot', 0.0)) + amount
        elif kind == 'scale':
            # multiplicative scale adjustment with clamping
            cur = float(a.get('scale', 1.0))
            # avoid negative or zero multiplier
            mul = max(0.1, (1.0 + amount))
            cur = cur * mul
            # clamp to safe bounds
            SCALE_MIN = 0.05
            SCALE_MAX = 10.0
            a['scale'] = max(SCALE_MIN, min(cur, SCALE_MAX))
        # Once adjusted, disable autoscale and persist
        self.disable_autoscale()
        self.save_anchors()

    def snap_pivot_to_joint(self, bone_name: str):
        """移动贴图位置（修改 dx/dy）使得自定义 pivot 与该骨骼关节重合。
        数学上 pivot 世界坐标 = (origin + 骨骼世界位置) + 旋转后的 (dx, dy)；
        因此令 dx=0, dy=0 即可保证无论骨骼角度与翻转如何，pivot 恒等于关节位置。
        """
        a = self.get_anchor(bone_name)
        a['dx'] = 0.0
        a['dy'] = 0.0
        self.disable_autoscale()
        self.save_anchors()

    def snap_all_pivots_to_joint(self):
        """对所有已加载贴图，将 pivot 对齐到各自骨骼关节（dx=dy=0）。"""
        for b in self.puppet.bones():
            name = b.name
            if name in self.images:
                a = self.get_anchor(name)
                a['dx'] = 0.0
                a['dy'] = 0.0
        self.disable_autoscale()
        self.save_anchors()

    def _sync_counterpart_scales(self):
        base = os.path.basename(self.anchors_path)
        if not base.startswith('anchors_') or not base.endswith('.json'):
            return
        name = base[len('anchors_'):-len('.json')]
        if name.endswith('_flip'):
            role = name[:-len('_flip')]
            counterpart = f'anchors_{role}.json'
        else:
            role = name
            counterpart = f'anchors_{role}_flip.json'
        cpath = os.path.join(os.path.dirname(self.anchors_path), counterpart)
        if not os.path.exists(cpath):
            return
        try:
            with open(cpath, 'r', encoding='utf-8') as f:
                cdata = json.load(f)
        except Exception:
            return
        cans = cdata.get('anchors', {})
        for k, v in self.anchors.items():
            if k in cans:
                cans[k]['scale'] = v.get('scale', cans[k].get('scale', 1.0))
        cdata['anchors'] = cans
        with open(cpath, 'w', encoding='utf-8') as f:
            json.dump(cdata, f, ensure_ascii=False, indent=2)
