import math


class Vec2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def rotated(self, deg: float):
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def to_tuple(self):
        return (self.x, self.y)


class Bone:
    def __init__(self, name: str, length: float, width: float, parent=None, local_offset: Vec2 = None, angle_deg: float = 0.0):
        self.name = name
        self.length = float(length)
        self.width = float(width)
        self.parent = parent
        self.local_offset = local_offset or Vec2(0, 0)
        self.angle_deg = float(angle_deg)
        self.children = []
        if parent:
            parent.children.append(self)

    def world_pos(self):
        if not self.parent:
            return self.local_offset
        # attach at parent's pivot, not tip
        base = self.parent.world_pos()
        # local offset rotated by parent's angle
        rotated = self.local_offset.rotated(self.parent.world_angle())
        return base + rotated

    def world_angle(self):
        if not self.parent:
            return self.angle_deg
        return self.parent.world_angle() + self.angle_deg

    def tip_offset(self):
        # tip along local +Y from the joint by length, rotated by world angle
        return Vec2(0, self.length).rotated(self.world_angle())


class GeomPuppet:
    def __init__(self, body_w: float = 80.0, body_h: float = 120.0):
        self.body_w = body_w
        self.body_h = body_h

        # Body as root (match original demo)
        self.body = Bone('body', length=body_h, width=body_w, parent=None, local_offset=Vec2(0, 0), angle_deg=0)
        
        # Head & waist (fixed sizes like original)
        self.head = Bone('head', length=60, width=60, parent=self.body, local_offset=Vec2(0, -body_h / 2), angle_deg=0)
        waist_start_ratio = 0.25
        waist_end_y = body_h * 0.35 + 90.0
        waist_len = max(10.0, waist_end_y - body_h * waist_start_ratio)
        self.waist = Bone('waist', length=waist_len, width=60, parent=self.body, local_offset=Vec2(0, body_h * waist_start_ratio), angle_deg=0)

        # Arms: attach at body centerline, opened pose
        shoulder_drop = 20
        shoulder_y = -body_h / 2 + shoulder_drop
        upper_len = 70
        lower_len = 60

        # 向下45°张开作为默认姿势
        self.left_upper_arm = Bone('left_upper_arm', length=upper_len, width=20, parent=self.body, local_offset=Vec2(0, shoulder_y), angle_deg=-45.0)
        self.left_lower_arm = Bone('left_lower_arm', length=lower_len, width=16, parent=self.left_upper_arm, local_offset=Vec2(0, upper_len), angle_deg=0.0)

        self.right_upper_arm = Bone('right_upper_arm', length=upper_len, width=20, parent=self.body, local_offset=Vec2(0, shoulder_y), angle_deg=45.0)
        self.right_lower_arm = Bone('right_lower_arm', length=lower_len, width=16, parent=self.right_upper_arm, local_offset=Vec2(0, upper_len), angle_deg=0.0)

        # Legs attach to waist bottom
        pelvis_w = self.waist.width
        pelvis_h = self.waist.length
        leg_attach_y = pelvis_h - 6.0

        self.left_leg = Bone('left_leg', length=90, width=22, parent=self.waist, local_offset=Vec2(-pelvis_w * 0.6, leg_attach_y), angle_deg=0.0)
        self.right_leg = Bone('right_leg', length=90, width=22, parent=self.waist, local_offset=Vec2(pelvis_w * 0.6, leg_attach_y), angle_deg=0.0)

    def bones(self):
        # Render order to match original demo's ordered_parts
        return [
            self.waist,
            self.body,
            self.head,
            self.left_leg,
            self.right_leg,
            self.left_upper_arm,
            self.left_lower_arm,
            self.right_upper_arm,
            self.right_lower_arm,
        ]

    def reset_pose(self):
        # 重置为初始化姿态：躯干/头/髋归零，手臂向下45°张开，前臂与腿为0°
        self.body.angle_deg = 0.0
        self.head.angle_deg = 0.0
        self.waist.angle_deg = 0.0
        self.left_upper_arm.angle_deg = -45.0
        self.right_upper_arm.angle_deg = 45.0
        self.left_lower_arm.angle_deg = 0.0
        self.right_lower_arm.angle_deg = 0.0
        self.left_leg.angle_deg = 0.0
        self.right_leg.angle_deg = 0.0