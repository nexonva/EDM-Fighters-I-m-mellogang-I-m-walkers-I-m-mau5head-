# -*- coding: utf-8 -*-
"""
EDM Fighter (Pygame, single file) — Full Integrated Build (audio removed)

Balance Patch v1.1 — 2025-09-01

Patch notes inline as comments below so you can tweak easily.

Fan-made, non-commercial parody project. Parody-only visuals, no logos.
"""

import pygame
import random
import math
from collections import deque

# ------------------------------
# Basic setup
# ------------------------------
pygame.init()
# audio removed: no pygame.mixer.init()
pygame.display.set_caption("EDM Fighter")

W, H = 1280, 720
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (240, 240, 240)
BLACK = (10, 10, 10)
GRAY  = (60, 60, 60)
RED   = (235, 70, 70)
GREEN = (60, 220, 120)
BLUE  = (80, 160, 255)
YELLOW= (255, 210, 80)
PURPLE= (180, 120, 255)
CYAN  = (120, 255, 255)
ORANGE= (255, 160, 90)
PINK  = (255, 140, 210)
TAN   = (210, 180, 120)

# Deadmau5 darker red + neon hues
DEAD_RED   = (170, 40, 40)
NEON_CYAN  = (80, 255, 255)
NEON_MAG   = (255, 80, 200)

FONT_BIG   = pygame.font.SysFont("arial", 48)
FONT_MED   = pygame.font.SysFont("arial", 28)
FONT_SMALL = pygame.font.SysFont("arial", 20)

GRAVITY = 0.9

# ------------------------------
# Global Balance Knobs (easy to tweak)
# ------------------------------
METER_GAIN_LIGHT = 5         # was 7
METER_GAIN_SPECIAL = 8       # was 10
METER_GAIN_ON_HIT_OWNER = 9  # was 12
METER_GAIN_ON_HIT_VICTIM = 6 # was 8

GUARD_DMG_MULT = 0.60        # was 0.55 (blocking takes slightly more dmg)
SHIELD_DMG_MULT = 0.70       # was 0.65 (Marshmello shield reduces a bit less)
GUARD_KB_MULT = 0.50         # was 0.40 (more push on block -> resets to neutral more)

# ------------------------------
# Helpers
# ------------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_bar(x, y, w, h, ratio, back_color=(50,50,50), fill_color=(200,50,50), border=True):
    pygame.draw.rect(screen, back_color, (x, y, w, h), border_radius=6)
    fw = int(w * clamp(ratio, 0, 1))
    pygame.draw.rect(screen, fill_color, (x, y, fw, h), border_radius=6)
    if border:
        pygame.draw.rect(screen, (20,20,20), (x, y, w, h), 2, border_radius=6)

def sign(x): 
    return -1 if x < 0 else (1 if x > 0 else 0)

def text_center(text, font, color, pos):
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=pos)
    screen.blit(surf, rect)

def draw_glow_circle(center, base_radius, color, layers=4, max_alpha=120):
    """Simple neon glow: stacked translucent circles."""
    cx, cy = center
    for i in range(layers, 0, -1):
        r = int(base_radius * (1 + 0.25 * i))
        alpha = max(0, int(max_alpha * (i / (layers + 1))))
        s = pygame.Surface((r*2+2, r*2+2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (r+1, r+1), r, width=2)
        screen.blit(s, (cx - r - 1, cy - r - 1))

# ------------------------------
# Attack / Projectile
# ------------------------------
class Attack:
    def __init__(self, owner, rect, dmg=10, kb=(6, -4), duration=12, 
                 pierce=False, projectile=False, v=(0,0), bounces=0, tag=""):
        self.owner = owner
        self.rect = pygame.Rect(rect)
        self.dmg = dmg
        self.kb = kb
        self.t = duration
        self.pierce = pierce
        self.projectile = projectile
        self.vx, self.vy = v
        self.bounces = bounces
        self.hit_targets = set()
        self.tag = tag  # for special handling

    def update(self, world):
        if self.projectile:
            # Move
            self.rect.x += int(self.vx)
            self.vy += GRAVITY * 0.25  # light gravity for projectiles
            self.rect.y += int(self.vy)
            # floor bounce
            floor = world.floor_y
            if self.rect.bottom >= floor:
                if self.bounces > 0:
                    self.rect.bottom = floor
                    self.vy = -abs(self.vy) * 0.7
                    self.bounces -= 1
                else:
                    # stop vertical
                    self.rect.bottom = floor
                    self.vy = 0
            # boomerang slow return
            if self.tag == "boomer":
                self.vx += -sign(self.vx) * 0.2
        self.t -= 1
        return self.t <= 0

    def draw(self, world):
        # visuals by tag
        if self.tag == "cube_bass":
            # neon cube with glow
            pygame.draw.rect(screen, PINK, self.rect, border_radius=6)
            pygame.draw.rect(screen, (40, 0, 30), self.rect, 2, border_radius=6)
            draw_glow_circle(self.rect.center, max(self.rect.w, self.rect.h)//2, NEON_MAG, layers=3, max_alpha=110)
        elif self.tag == "chain":
            pygame.draw.rect(screen, (180, 180, 180), self.rect, border_radius=2)
        elif self.tag == "boomer":
            pygame.draw.rect(screen, CYAN, self.rect, border_radius=6)
        elif self.tag == "dub":
            pygame.draw.circle(screen, BLUE, self.rect.center, max(8, self.rect.w//2), width=2)
            pygame.draw.circle(screen, (30,60,120), self.rect.center, max(3, self.rect.w//4))
        else:
            c = ORANGE if self.projectile else YELLOW
            pygame.draw.rect(screen, c, self.rect, border_radius=4)

# ------------------------------
# Fighter
# ------------------------------
class Fighter:
    def __init__(self, name, x, y, color=(200,200,200), human=True):
        self.name = name
        self.rect = pygame.Rect(x, y, 56, 80)
        self.color = color
        self.human = human
        self.face = 1
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.hp = 150
        self.max_hp = 150
        self.meter = 0
        self.max_meter = 100
        self.hitstun = 0
        self.invuln = 0
        self.guard = False
        self.guard_effect = 0

        # cooldowns
        self.cd_light = 0
        self.cd_special = 0
        self.cd_ult = 0
        self.action_lock = 0  # for dash etc.

        # status effects (seconds -> frames)
        self.regen_t = 0
        self.regen_rate = 0.7  # hp per frame while regen active
        self.shield_t = 0
        self.slow_aura_t = 0
        self.slowed_t = 0
        self.clone_t = 0

        # ghost mirror for Chainsmokers
        self.ghost_cool = 0

        # particles (for The Chainsmokers smoke)
        self.smoke = deque(maxlen=80)
        self.smoke_tick = 0

        # AI
        self.ai_state = {"decide_t": 0, "want": "approach", "jump_t": 0, "rng": 0}

        # visuals
        self.fx = deque(maxlen=80)

        # flavor: base move parameters (tunable per character)
        self.base_speed = 5.2
        self.jump_power = -18

    def speed_mod(self):
        mod = 1.0
        if self.slowed_t > 0:
            mod *= 0.65
        if self.shield_t > 0 and self.name == "Marshmello":
            mod *= 0.82  # was 0.85 (shield slows self a bit more)
        return mod

    def can_act(self):
        return self.hitstun <= 0 and self.action_lock <= 0

    def gain_meter(self, amount):
        self.meter = clamp(self.meter + amount, 0, self.max_meter)

    def take_damage(self, dmg, src=None):
        if self.invuln > 0:
            return  # no damage
        # guard reduces
        if self.guard:
            dmg *= GUARD_DMG_MULT
        # shield reduces more
        if self.shield_t > 0:
            dmg *= SHIELD_DMG_MULT
        self.hp -= int(max(1, dmg))

    def apply_knockback(self, kb, from_dir=1):
        kx, ky = kb
        if self.guard:
            kx *= GUARD_KB_MULT
            ky *= GUARD_KB_MULT
        self.vx += kx * from_dir
        self.vy += ky

    def start_hitstun(self, frames=12):
        if self.guard:
            self.hitstun = max(self.hitstun, int(frames * 0.4))
        else:
            self.hitstun = max(self.hitstun, frames)

    def update(self, input_state, world, opponent):
        # timers
        for attr in ["cd_light","cd_special","cd_ult","hitstun","invuln","action_lock","guard_effect",
                     "regen_t","shield_t","slow_aura_t","slowed_t","clone_t","ghost_cool"]:
            v = getattr(self, attr)
            if v > 0: setattr(self, attr, v-1)

        # regen
        if self.regen_t > 0:
            self.hp = clamp(self.hp + self.regen_rate, 0, self.max_hp)

        # slow aura (Marshmello ultimate)
        if self.slow_aura_t > 0 and abs(self.rect.centerx - opponent.rect.centerx) < 220 and abs(self.rect.centery - opponent.rect.centery) < 160:
            opponent.slowed_t = max(opponent.slowed_t, 10)

        # input / AI
        left = right = up = down = att = sp = ul = False
        if self.human:
            left = input_state.get("left", False)
            right= input_state.get("right", False)
            up   = input_state.get("up", False)
            down = input_state.get("down", False)
            att  = input_state.get("light", False)
            sp   = input_state.get("special", False)
            ul   = input_state.get("ult", False)
        else:
            self.ai_control(opponent, world)
            left, right, up, down, att, sp, ul = self.ai_buttons

        # facing
        self.face = -1 if self.rect.centerx > opponent.rect.centerx else 1

        # actions + stage physics modifiers
        spd = self.base_speed * self.speed_mod()
        ground_friction = world.stage.ground_friction
        air_drag = world.stage.air_drag

        if self.can_act():
            self.guard = down
            if left and not right:
                self.vx = -spd
            elif right and not left:
                self.vx = spd
            else:
                self.vx *= ground_friction if self.on_ground else air_drag

            if up and self.on_ground:
                self.vy = self.jump_power
                self.on_ground = False

            # attacks
            if att and self.cd_light <= 0:
                self.light_attack(world, opponent)
            if sp and self.cd_special <= 0:
                self.special(world, opponent)
            if ul and self.cd_ult <= 0 and self.meter >= self.max_meter:
                self.ultimate(world, opponent)
        else:
            self.vx *= 0.9
            self.guard = False

        # physics
        self.vy += GRAVITY
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)

        # world bounds & floor
        if self.rect.bottom >= world.floor_y:
            self.rect.bottom = world.floor_y
            self.vy = 0
            self.on_ground = True
        else:
            self.on_ground = False

        self.rect.left = clamp(self.rect.left, 40, W-40-self.rect.width)

        # Alan Walker dash trail
        if self.action_lock > 0 and self.name in ("Alan Walker",):
            self.fx.append((self.rect.copy(), 10))

        # --- The Chainsmokers: cigarette-like smoke particles ---
        if self.name == "The Chainsmokers":
            self.smoke_tick += 1
            if self.smoke_tick % 6 == 0:
                cx, cy = self.rect.center
                tip1 = (cx - 8, cy - 6)
                tip2 = (cx + 8, cy - 10)
                for (sx, sy) in (tip1, tip2):
                    self.smoke.append([sx, sy, 2.0, 40])
            for p in list(self.smoke):
                p[1] -= 0.8
                p[2] += 0.08
                p[3] -= 1
                if p[3] <= 0:
                    self.smoke.remove(p)

        # decay fx lifetimes
        for i in range(len(self.fx)):
            r, t = self.fx[i]
            self.fx[i] = (r, t-1)
        while self.fx and self.fx[0][1] <= 0:
            self.fx.popleft()

    # ------------- Attacks -------------
    def light_attack(self, world, opponent):
        self.cd_light = 18
        self.action_lock = 8
        w, h = 44, 36
        ox = 30 * self.face
        hit = Attack(self, (self.rect.centerx + ox - w//2, self.rect.centery - h//2, w, h),
                     dmg=10, kb=(7, -6), duration=8)
        world.attacks.append(hit)
        self.gain_meter(METER_GAIN_LIGHT)

        # Chainsmokers ghost mirror
        if self.clone_t > 0 and self.ghost_cool <= 0:
            self.ghost_cool = 12
            g = Attack(self, (self.rect.centerx + ox*1.4 - w//2, self.rect.centery - h//2, int(w*0.9), int(h*0.9)),
                       dmg=4, kb=(5, -5), duration=8)  # dmg was 5
            world.attacks.append(g)

    def special(self, world, opponent):
        self.cd_special = 60
        self.action_lock = 10
        self.gain_meter(METER_GAIN_SPECIAL)
        if self.name == "Kygo":
            self.hp = clamp(self.hp + 6, 0, self.max_hp)  # was 10
            pulse = Attack(self, (self.rect.centerx-40, self.rect.centery-40, 80, 80),
                           dmg=7, kb=(6, -5), duration=10)  # dmg was 8
            world.attacks.append(pulse)
            world.spawn_ring(self.rect.center, 80, GREEN)

        elif self.name == "Marshmello":
            self.shield_t = int(1.5*FPS)  # was 2*FPS
            world.spawn_ring(self.rect.center, 70, WHITE)

        elif self.name == "Alan Walker":
            self.invuln = 10   # was 14
            self.action_lock = 12
            self.vx = 14 * self.face  # was 16
            stab = Attack(self, (self.rect.centerx + 20*self.face - 18, self.rect.centery-18, 36, 36),
                          dmg=12, kb=(9, -7), duration=10)  # dmg was 14
            world.attacks.append(stab)
            world.spawn_shadow(self)

        elif self.name == "Deadmau5":
            bx = self.rect.centerx + 30*self.face
            by = self.rect.centery - 10
            proj = Attack(self, (bx, by, 26, 26), dmg=10, kb=(7, -6), duration=80,  # dmg was 12
                          pierce=False, projectile=True, v=(9*self.face, -3), bounces=0, tag="boomer")  # vx was 10
            world.attacks.append(proj)
            world.spawn_ring(self.rect.center, 60, NEON_CYAN)

        elif self.name == "The Chainsmokers":
            chain = Attack(self, (self.rect.centerx + 18*self.face, self.rect.centery-12, 60, 24),
                           dmg=10, kb=(-3, -3), duration=10, tag="chain")
            world.attacks.append(chain)
            world.spawn_chain(self.rect.center, opponent.rect.center)

        elif self.name == "Fisher":
            # surf dash (logic) + fishing-rod visual handled in draw()
            self.invuln = 6  # was 8
            self.vx = 13 * self.face  # was 14
            surf = Attack(self, (self.rect.centerx + 24*self.face, self.rect.centery-20, 48, 40),
                          dmg=12, kb=(9, -6), duration=12)  # dmg was 14
            world.attacks.append(surf)
            world.spawn_ring(self.rect.center, 60, ORANGE)

        elif self.name == "Ben UFO":
            proj = Attack(self, (self.rect.centerx + 28*self.face, self.rect.centery-12, 24, 24),
                          dmg=10, kb=(6,-5), duration=60,  # dmg was 11
                          projectile=True, v=(8*self.face, -2), bounces=1, tag="dub")
            world.attacks.append(proj)

    def ultimate(self, world, opponent):
        self.cd_ult = 120
        self.meter = 0
        self.action_lock = 18
        if self.name == "Kygo":
            self.hp = clamp(self.hp + 18, 0, self.max_hp)  # was 30
            self.regen_t = int(4*FPS)  # was 5*FPS
            wave = Attack(self, (self.rect.centerx-120, self.rect.centery-120, 240, 240),
                          dmg=16, kb=(10, -9), duration=18)  # dmg was 18
            world.attacks.append(wave)
            world.spawn_ring(self.rect.center, 140, GREEN, bold=True)

        elif self.name == "Marshmello":
            self.invuln = int(0.4*FPS)  # was 0.5*FPS
            self.slow_aura_t = int(2.5*FPS)  # was 3*FPS
            world.spawn_ring(self.rect.center, 180, WHITE, bold=True)

        elif self.name == "Alan Walker":
            opponent.slowed_t = max(opponent.slowed_t, int(2.5*FPS))  # was 3*FPS
            for i in range(5):
                stab = Attack(self, (self.rect.centerx + (30+18*i)*self.face - 18, self.rect.centery-18, 36, 36),
                              dmg=5, kb=(7, -6), duration=12)  # dmg was 6
                world.attacks.append(stab)
            self.invuln = 14  # was 20
            self.vx = 16*self.face  # was 18
            self.action_lock = 20
            world.spawn_shadow(self, strong=True)

        elif self.name == "Deadmau5":
            bx = self.rect.centerx + 30*self.face
            by = self.rect.centery - 30
            cube = Attack(self, (bx, by, 48, 48), dmg=22, kb=(12, -10), duration=200,  # dmg was 26, dur was 240
                          pierce=False, projectile=True, v=(11*self.face, -6), bounces=2, tag="cube_bass")  # vx 12->11, bounces 3->2
            world.attacks.append(cube)

        elif self.name == "The Chainsmokers":
            self.clone_t = int(5*FPS)  # was 6*FPS
            world.spawn_clone(self)

        elif self.name == "Fisher":
            wave = Attack(self, (self.rect.centerx-100, self.rect.centery-80, 200, 160),
                          dmg=18, kb=(12,-8), duration=20)  # dmg was 20
            world.attacks.append(wave)
            world.spawn_ring(self.rect.center, 160, ORANGE, bold=True)

        elif self.name == "Ben UFO":
            opponent.slowed_t = max(opponent.slowed_t, int(2.5*FPS))  # was 3*FPS
            for i in range(-2, 3):
                proj = Attack(self, (self.rect.centerx + 20*i, self.rect.centery-20, 20, 20),
                              dmg=6, kb=(5,-4), duration=70,  # dmg was 7
                              projectile=True, v=(6*i, -3), bounces=0, tag="dub")
                world.attacks.append(proj)
            world.spawn_ring(self.rect.center, 200, BLUE, bold=True)

    # ------------- AI -------------
    def ai_control(self, opponent, world):
        if self.ai_state["decide_t"] <= 0:
            self.ai_state["decide_t"] = random.randint(8, 14)
            dist = opponent.rect.centerx - self.rect.centerx
            self.ai_state["rng"] = random.random()
            if abs(dist) > 420:
                self.ai_state["want"] = "approach"
            elif abs(dist) < 120:
                self.ai_state["want"] = "retreat" if random.random() < 0.35 else "attack"
            else:
                self.ai_state["want"] = "mid"
            if self.on_ground and random.random() < 0.12:
                self.ai_state["jump_t"] = 3
        else:
            self.ai_state["decide_t"] -= 1

        left = right = up = down = att = sp = ul = False
        want = self.ai_state["want"]
        dist = opponent.rect.centerx - self.rect.centerx

        # Slightly more conservative specials on Easy/Normal
        aggro = [0.6, 0.75, 0.9][world.cpu_mode]
        special_rate = [0.05, 0.10, 0.18][world.cpu_mode]  # was 0.06/0.12/0.2
        guard_rate = [0.06, 0.12, 0.18][world.cpu_mode]

        if want == "approach":
            right = dist > 0
            left = dist < 0
            att = abs(dist) < 120 and random.random() < aggro
            sp = 120 < abs(dist) < 360 and random.random() < special_rate
        elif want == "retreat":
            right = dist < 0
            left = dist > 0
            down = random.random() < guard_rate
            att = abs(dist) < 120 and random.random() < 0.35
        elif want == "mid":
            if abs(dist) < 200:
                right = dist < 0 and random.random() < 0.6
                left  = dist > 0 and random.random() < 0.6
            else:
                right = dist > 0
                left  = dist < 0
            sp = random.random() < special_rate
            att = abs(dist) < 150 and random.random() < aggro*0.7

        if self.ai_state["jump_t"] > 0:
            up = True
            self.ai_state["jump_t"] -= 1

        if self.meter >= self.max_meter and random.random() < (0.015 + 0.02*world.cpu_mode):
            ul = True

        self.ai_buttons = (left, right, up, down, att, sp, ul)

    # ------------- Draw -------------
    def draw(self, world):
        # body
        body = self.rect.copy()
        pygame.draw.rect(screen, self.color, body, border_radius=10)
        pygame.draw.rect(screen, BLACK, body, 2, border_radius=10)

        cx, cy = self.rect.center

        if self.name == "Marshmello":
            head = pygame.Rect(0,0,46,46); head.center=(cx, cy-28)
            pygame.draw.rect(screen, WHITE, head, border_radius=8)
            def draw_x_eye(x,y):
                pygame.draw.line(screen, BLACK, (x-6,y-6),(x+6,y+6), 3)
                pygame.draw.line(screen, BLACK, (x+6,y-6),(x-6,y+6), 3)
            draw_x_eye(head.centerx-10, head.centery-4)
            draw_x_eye(head.centerx+10, head.centery-4)
            pygame.draw.line(screen, BLACK, (head.centerx-9, head.centery+11),(head.centerx+9, head.centery+11),2)

        elif self.name == "Alan Walker":
            hood_pts = [(cx-20, cy-34),(cx, cy-56),(cx+20, cy-34)]
            pygame.draw.polygon(screen, (70,70,85), hood_pts)
            pygame.draw.rect(screen, (200,200,210), (cx-10, cy-42, 20, 16), border_radius=4)
            c1 = [(cx-14, cy-6),(cx, cy-18),(cx+14, cy-6)]
            c2 = [(cx-12, cy+2),(cx, cy-10),(cx+12, cy+2)]
            pygame.draw.lines(screen, (120,130,170), False, c1, 3)
            pygame.draw.lines(screen, (120,130,170), False, c2, 3)

        elif self.name == "Deadmau5":
            head_r = 18
            head_center = (cx, cy-30)
            ear_r = 12
            left_ear = (cx-20, cy-44)
            right_ear= (cx+20, cy-44)
            pygame.draw.circle(screen, DEAD_RED, head_center, head_r)
            pygame.draw.circle(screen, DEAD_RED, left_ear, ear_r)
            pygame.draw.circle(screen, DEAD_RED, right_ear, ear_r)
            pygame.draw.circle(screen, BLACK, (cx-5, cy-32), 2)
            pygame.draw.circle(screen, BLACK, (cx+5, cy-32), 2)
            pygame.draw.rect(screen, BLACK, (cx-6, cy-26, 12, 2), border_radius=1)
            # neon glow
            t = pygame.time.get_ticks() * 0.005
            pulse = 1.0 + 0.15 * math.sin(t)
            base_alpha = 110
            draw_glow_circle(head_center, int(head_r*pulse), NEON_CYAN, layers=3, max_alpha=base_alpha)
            draw_glow_circle(left_ear,  int(ear_r*pulse), NEON_CYAN, layers=2, max_alpha=base_alpha-20)
            draw_glow_circle(right_ear, int(ear_r*pulse), NEON_CYAN, layers=2, max_alpha=base_alpha-20)
            draw_glow_circle(head_center, int((head_r-2)*pulse), NEON_MAG, layers=2, max_alpha=80)

        elif self.name == "The Chainsmokers":
            def draw_cig(center, angle_deg, length=36, thickness=6):
                lx = math.cos(math.radians(angle_deg))
                ly = math.sin(math.radians(angle_deg))
                half = length/2
                x0, y0 = center[0] - half*lx, center[1] - half*ly
                x1, y1 = center[0] + half*lx, center[1] + half*ly
                pygame.draw.line(screen, (215,215,215), (x0,y0), (x1,y1), thickness)
                fx, fy = center[0] - (half-6)*lx, center[1] - (half-6)*ly
                pygame.draw.line(screen, TAN, (fx-6*lx, fy-6*ly), (fx, fy), thickness)
                tx, ty = center[0] + half*lx, center[1] + half*ly
                pygame.draw.circle(screen, ORANGE, (int(tx), int(ty)), 4)
            draw_cig((cx-6, cy-2), -25)
            draw_cig((cx+6, cy-10), 155)
            for p in list(self.smoke):
                x, y, r, life = p
                alpha = max(0, min(140, int(60 + life*2)))
                s = pygame.Surface((int(r*4), int(r*4)), pygame.SRCALPHA)
                pygame.draw.circle(s, (200,200,200, alpha), (int(r*2), int(r*2)), int(r*2))
                screen.blit(s, (x - int(r*2), y - int(r*2)))

        elif self.name == "Kygo":
            pygame.draw.circle(screen, YELLOW, (cx, cy-10), 11)
            pygame.draw.line(screen, (90,70,40), (cx, cy-2), (cx, cy+8), 3)
            pygame.draw.line(screen, GREEN, (cx, cy-2), (cx-8, cy-6), 2)
            pygame.draw.line(screen, GREEN, (cx, cy-2), (cx+8, cy-6), 2)

        elif self.name == "Fisher":
            pygame.draw.circle(screen, ORANGE, (cx, cy-28), 20)
            pygame.draw.arc(screen, BLACK, (cx-12, cy-36, 24, 20), 3.4, 6.0, 2)
            pygame.draw.circle(screen, BLACK, (cx-6, cy-32), 3)
            pygame.draw.circle(screen, BLACK, (cx+6, cy-32), 3)
            # Fishing rod (diagonal)
            hx = cx + (12 * self.face)
            hy = cy - 6
            tipx = hx + 72 * self.face
            tipy = hy - 34
            pygame.draw.line(screen, (110, 80, 40), (hx, hy), (tipx, tipy), 5)
            pygame.draw.circle(screen, (70, 60, 50), (hx-6*self.face, hy+8), 6)
            pygame.draw.line(screen, (220, 220, 230), (tipx, tipy), (tipx + 10*self.face, tipy + 20), 2)
            hook_base = (tipx + 10*self.face, tipy + 20)
            pygame.draw.arc(screen, (220,220,230),
                            (hook_base[0]-6, hook_base[1]-8, 12, 14),
                            0.6 if self.face==1 else 2.6,
                            4.7 if self.face==1 else 6.7, 2)

        elif self.name == "Ben UFO":
            hood = [(cx-20, cy-30),(cx, cy-55),(cx+20, cy-30)]
            pygame.draw.polygon(screen, (40,40,60), hood)
            pygame.draw.circle(screen, (20,20,30), (cx, cy-40), 10)
            pygame.draw.circle(screen, BLUE, (cx, cy), 18, width=2)

        # guard/Shield tint
        if self.guard or self.shield_t>0:
            g = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            pygame.draw.rect(g, (120,200,255, 80 if self.guard else 110), (0,0,*g.get_size()), border_radius=10)
            screen.blit(g, self.rect.topleft)

        if self.slow_aura_t > 0 and self.name == "Marshmello":
            pygame.draw.circle(screen, WHITE, (cx, cy), 180, width=2)

        if self.clone_t > 0 and self.name == "The Chainsmokers":
            r = self.rect.inflate(10, 10)
            pygame.draw.rect(screen, PINK, r, 2, border_radius=12)

        # Alan Walker dash shadows
        for r, t in self.fx:
            alpha = max(0, min(120, t*6))
            surf = pygame.Surface((r.width, r.height), pygame.SRCALPHA)
            pygame.draw.rect(surf, (100,100,160, alpha), (0,0,r.width,r.height), border_radius=10)
            screen.blit(surf, r.topleft)

# ------------------------------
# Stage System
# ------------------------------
class Stage:
    def __init__(self, name, floor_y, bg_kind="neon", ground_friction=0.75, air_drag=0.95):
        self.name = name
        self.floor_y = floor_y
        self.bg_kind = bg_kind
        self.ground_friction = ground_friction
        self.air_drag = air_drag
        # stage timers/effects
        self.ticks = 0
        self.wave_timer = 0  # for beach
        self.spot_angle = 0  # for club

    def update(self):
        self.ticks += 1
        if self.bg_kind == "beach":
            self.wave_timer = (self.wave_timer + 1) % (6*FPS)
        elif self.bg_kind == "club":
            self.spot_angle = (self.spot_angle + 0.01) % (2*math.pi)

    def draw_background(self):
        if self.bg_kind == "neon":
            for i in range(H):
                c = (20 + i//12, 20 + i//10, 38 + i//8)
                pygame.draw.line(screen, c, (0,i),(W,i))
            for k in range(4):
                x = (self.ticks*2 + k*280) % (W+600) - 300
                poly = [(x-200,0),(x+200,0),(x+400,H),(x,H)]
                s = pygame.Surface((W,H), pygame.SRCALPHA)
                pygame.draw.polygon(s, (120, 200, 255, 18), poly)
                screen.blit(s,(0,0))
        elif self.bg_kind == "beach":
            for i in range(H):
                c = (40 + i//16, 80 + i//18, 120 + i//20)
                pygame.draw.line(screen, c, (0,i),(W,i))
            pygame.draw.circle(screen, (255,230,120), (W-160, 140), 60)
            for j in range(5):
                y = int(self.floor_y - 60 - j*22 + 6*math.sin((self.ticks*0.05)+j))
                pygame.draw.line(screen, (200,220,240), (0,y),(W,y), 2)
        elif self.bg_kind == "club":
            screen.fill((12, 10, 18))
            for k, col in enumerate([(180,60,200,50),(60,180,220,50)]):
                ang = self.spot_angle + (k*math.pi/3)
                x0, y0 = W//2, 100
                x1 = int(x0 + 900*math.cos(ang))
                y1 = int(y0 + 900*math.sin(ang))
                s = pygame.Surface((W,H), pygame.SRCALPHA)
                pygame.draw.polygon(s, col, [(x0,y0),(x1,y1),(x1+200,y1)], 0)
                screen.blit(s,(0,0))

    def stage_fx(self, world, fighters):
        if self.bg_kind == "beach":
            if self.wave_timer == 0:
                for f in fighters:
                    world.spawn_ring(f.rect.center, 110, (200,220,240))
                    f.vx *= 0.95

# ------------------------------
# World / Stage Wrapper
# ------------------------------
class World:
    def __init__(self, stage: Stage):
        self.stage = stage
        self.floor_y = stage.floor_y
        self.attacks = []
        self.cpu_mode = 1  # 0 easy, 1 normal, 2 hard
        self.fx_rings = []
        self.fx_chains = []
        self.fx_shadows = []

    def update(self, fighters):
        self.stage.update()
        self.floor_y = self.stage.floor_y

        # attacks
        for atk in list(self.attacks):
            dead = atk.update(self)
            for f in fighters:
                if f is atk.owner: 
                    continue
                if atk.rect.colliderect(f.rect) and f not in atk.hit_targets:
                    atk.hit_targets.add(f)
                    dmg = atk.dmg
                    f.take_damage(dmg, src=atk.owner)
                    atk.owner.gain_meter(METER_GAIN_ON_HIT_OWNER)
                    f.gain_meter(METER_GAIN_ON_HIT_VICTIM)
                    dir = sign(atk.owner.rect.centerx - f.rect.centerx)
                    f.apply_knockback((-atk.kb[0], atk.kb[1]), from_dir=dir)
                    f.start_hitstun(12)
            if dead:
                self.attacks.remove(atk)

        # fx lifetimes
        self.fx_rings = [(pos, r, c, t-1, bold) for (pos, r, c, t, bold) in self.fx_rings if t-1>0]
        self.fx_chains = [(a,b,t-1) for (a,b,t) in self.fx_chains if t-1>0]
        self.fx_shadows = [(f,t-1,s) for (f,t,s) in self.fx_shadows if t-1>0]

        # ambient stage effects
        self.stage.stage_fx(self, fighters)

    def draw(self, fighters):
        self.stage.draw_background()
        pygame.draw.rect(screen, (30,30,30), (0, self.floor_y, W, H-self.floor_y))
        for x in range(0, W, 80):
            pygame.draw.rect(screen, (40,40,40), (x, self.floor_y, 40, 6))
        for (pos, r, c, t, bold) in self.fx_rings:
            width = 3 if not bold else 6
            pygame.draw.circle(screen, c, pos, int(r*(1+(60-t)/180)), width=width)
        for (a,b,t) in self.fx_chains:
            pygame.draw.line(screen, (200,200,200), a, b, 2)
        for (f,t,strong) in self.fx_shadows:
            r = f.rect.inflate(14,14)
            pygame.draw.rect(screen, (100,100,160), r, 2 if not strong else 4, border_radius=12)
        for atk in self.attacks:
            atk.draw(self)

    # fx helpers
    def spawn_ring(self, pos, radius, color, bold=False):
        self.fx_rings.append((pos, radius, color, 60, bold))
    def spawn_chain(self, a, b):
        self.fx_chains.append((a, b, 30))
    def spawn_shadow(self, f, strong=False):
        self.fx_shadows.append((f, 40, strong))
    def spawn_clone(self, f):
        pass

# ------------------------------
# NOTE: Audio removed entirely from this file.
# If you want to add audio back later, reintroduce pygame.mixer.init()
# and a minimal play function that checks for files and calls mixer.music.
# ------------------------------

# ------------------------------
# UI Helpers
# ------------------------------

def draw_hud(p1, p2, round_timer):
    text_center(p1.name, FONT_MED, WHITE, (150, 40))
    text_center(p2.name, FONT_MED, WHITE, (W-150, 40))
    draw_bar(40, 60, 460, 18, p1.hp/p1.max_hp, back_color=(60,25,25), fill_color=RED)
    draw_bar(W-40-460, 60, 460, 18, p2.hp/p2.max_hp, back_color=(60,25,25), fill_color=RED)
    draw_bar(40, 84, 460, 12, p1.meter/p1.max_meter, back_color=(20,35,55), fill_color=BLUE)
    draw_bar(W-40-460, 84, 460, 12, p2.meter/p2.max_meter, back_color=(20,35,55), fill_color=BLUE)
    text_center(f"{max(0, round_timer//FPS)}", FONT_BIG, WHITE, (W//2, 50))


def draw_select_menu(idx, names, cpu_mode):
    text_center("EDM Parody Fighter", FONT_BIG, WHITE, (W//2, 120))
    text_center("Character Select: A/D move, ENTER confirm, C CPU difficulty", FONT_MED, WHITE, (W//2, 170))
    text_center(f"CPU Difficulty: {['Easy','Normal','Hard'][cpu_mode]}", FONT_MED, YELLOW, (W//2, 210))
    cols = len(names)
    gap = W // (cols+1)
    for i, n in enumerate(names):
        x = (i+1)*gap; y = 340; w, h = 120, 160
        rect = pygame.Rect(x - w//2, y - h//2, w, h)
        c = (100,100,100) if i != idx else (140,160,200)
        pygame.draw.rect(screen, c, rect, border_radius=18)
        pygame.draw.rect(screen, BLACK, rect, 3, border_radius=18)
        text_center(n, FONT_SMALL, WHITE, (x, y+110))
        if i == idx:
            pygame.draw.rect(screen, YELLOW, rect.inflate(16,16), 3, border_radius=22)


def draw_stage_select(stage_idx, stages):
    text_center("Stage Select", FONT_BIG, WHITE, (W//2, 120))
    text_center("A/D to change stage, ENTER to confirm", FONT_MED, WHITE, (W//2, 170))
    names = [s.name for s in stages]
    text_center(f"{names[stage_idx]}", FONT_BIG, YELLOW, (W//2, 230))
    gap = 300
    for i, st in enumerate(stages):
        x = W//2 + (i-stage_idx)*gap
        y = 430
        w, h = 240, 120
        rect = pygame.Rect(x - w//2, y - h//2, w, h)
        pygame.draw.rect(screen, (90,90,120) if i!=stage_idx else (140,160,200), rect, border_radius=12)
        pygame.draw.rect(screen, BLACK, rect, 3, border_radius=12)
        text_center(st.name, FONT_SMALL, WHITE, (x, y+60))


def draw_result(winner_name):
    text_center(f"Winner: {winner_name}", FONT_BIG, WHITE, (W//2, H//2 - 20))
    text_center("ENTER: Play Again  |  ESC: Quit", FONT_MED, WHITE, (W//2, H//2 + 30))

# ------------------------------
# Game flow
# ------------------------------

def main():
    running = True
    state = "menu"  # "menu" -> "select_char" -> "select_stage" -> "fight" -> "result"

    # Stages
    stages = [
        Stage("Neon City", floor_y=int(H*0.82), bg_kind="neon", ground_friction=0.75, air_drag=0.95),
        Stage("Tropical Beach", floor_y=int(H*0.84), bg_kind="beach", ground_friction=0.90, air_drag=0.96),
        Stage("Underground Club", floor_y=int(H*0.80), bg_kind="club", ground_friction=0.76, air_drag=0.90),
    ]
    stage_idx = 0
    world = World(stages[stage_idx])

    names = ["Kygo","Marshmello","Alan Walker","Deadmau5","The Chainsmokers","Fisher","Ben UFO"]
    colors = {
        "Kygo": (240, 220, 160),
        "Marshmello": (220, 220, 220),
        "Alan Walker": (140, 140, 170),
        "Deadmau5": (170, 70, 70),  # darker body color
        "The Chainsmokers": (240, 160, 220),
        "Fisher": (255, 180, 120),
        "Ben UFO": (180, 200, 220),
    }
    sel_idx = 1
    p1 = p2 = None
    winner = None
    round_time = 99 * FPS

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # ------------- MENU -------------
        if state == "menu":
            screen.fill((18,22,30))
            text_center("EDM Parody Fighter", FONT_BIG, WHITE, (W//2, H//2 - 40))
            text_center("ENTER: Start  |  ESC: Quit", FONT_MED, WHITE, (W//2, H//2 + 10))
            if keys[pygame.K_RETURN]:
                state = "select_char"
                pygame.time.wait(160)
            if keys[pygame.K_ESCAPE]:
                running = False

        # ------------- CHARACTER SELECT -------------
        elif state == "select_char":
            world.stage.draw_background()  # backdrop
            draw_select_menu(sel_idx, names, world.cpu_mode)
            if keys[pygame.K_a]:
                sel_idx = (sel_idx - 1) % len(names)
            if keys[pygame.K_d]:
                sel_idx = (sel_idx + 1) % len(names)
            if keys[pygame.K_c]:
                world.cpu_mode = (world.cpu_mode + 1) % 3
                pygame.time.wait(160)
            if keys[pygame.K_RETURN]:
                state = "select_stage"
                pygame.time.wait(160)

        # ------------- STAGE SELECT -------------
        elif state == "select_stage":
            screen.fill((16,18,24))
            draw_stage_select(stage_idx, stages)
            if keys[pygame.K_a]:
                stage_idx = (stage_idx - 1) % len(stages)
                pygame.time.wait(130)
            if keys[pygame.K_d]:
                stage_idx = (stage_idx + 1) % len(stages)
                pygame.time.wait(130)
            if keys[pygame.K_RETURN]:
                world = World(stages[stage_idx])
                my_name = names[sel_idx]
                cpu_name = random.choice([n for n in names if n != my_name])
                p1 = Fighter(my_name, 300, 200, color=colors[my_name], human=True)
                p2 = Fighter(cpu_name, W-360, 200, color=colors[cpu_name], human=False)
                world.attacks.clear()
                world.fx_rings.clear()
                world.fx_chains.clear()
                world.fx_shadows.clear()
                winner = None
                state = "fight"
                round_timer = round_time
                pygame.time.wait(160)

        # ------------- FIGHT -------------
        elif state == "fight":
            world.update([p1, p2])

            # input mapping for P1
            inp = {
                "left": keys[pygame.K_a],
                "right": keys[pygame.K_d],
                "up": keys[pygame.K_w],
                "down": keys[pygame.K_s],
                "light": keys[pygame.K_j],
                "special": keys[pygame.K_k],
                "ult": keys[pygame.K_l],
            }
            p1.update(inp, world, p2)
            p2.update({}, world, p1)

            # resolve KOs / timer
            round_timer -= 1
            if p1.hp <= 0 or p2.hp <= 0 or round_timer <= 0:
                if p1.hp == p2.hp:
                    if int(p1.hp) == int(p2.hp):
                        winner = "Draw"
                    else:
                        winner = p1.name if p1.hp > p2.hp else p2.name
                else:
                    winner = p1.name if p1.hp > p2.hp else p2.name
                state = "result"

            # draw world & HUD
            world.draw([p1,p2])
            p1.draw(world)
            p2.draw(world)
            draw_hud(p1, p2, round_timer)

            # center divider & stage tag
            pygame.draw.line(screen, (30,30,60), (W//2, 120), (W//2, H), 1)
            text_center(stages[stage_idx].name, FONT_SMALL, WHITE, (W//2, 100))
            tip = "WASD move/jump, S guard, J Light, K Special, L Ultimate (Meter 100)"
            text_center(tip, FONT_SMALL, WHITE, (W//2, H-30))

        # ------------- RESULT -------------
        elif state == "result":
            screen.fill((18,22,30))
            draw_result(winner)

            if keys[pygame.K_RETURN]:
                state = "select_char"
                pygame.time.wait(160)
            if keys[pygame.K_ESCAPE]:
                running = False

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
