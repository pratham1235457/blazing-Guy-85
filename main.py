import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
from enum import Enum
import os


# ============================
# Advanced Harry Potter Theme
# ============================
class Spell(Enum):
    EXPECTO_PATRONUM = 1  # Main wand spell
    REDUCTO = 2  # Destructive spell (game over effect)
    LUMOS = 3  # Light spell (food glow)


class HarryPotterTheme:
    """
    Authentic Harry Potter color palette based on Hogwarts houses
    and medieval Gothic design principles
    """

    # GRYFFINDOR - Fire Element (Deep Crimson & Gold)
    GRYFFINDOR_DEEP_RED = (20, 0, 116)  # Deep crimson (BGR)
    GRYFFINDOR_SCARLET = (0, 0, 215)  # Bold scarlet
    GRYFFINDOR_GOLD = (0, 215, 255)  # Classic gold
    GRYFFINDOR_AMBER = (0, 140, 216)  # Firelight amber

    # SLYTHERIN - Water Element (Forest Green & Silver)
    SLYTHERIN_FOREST_GREEN = (34, 139, 34)  # Dark forest green
    SLYTHERIN_EMERALD = (0, 100, 100)  # Deep emerald
    SLYTHERIN_SILVER = (200, 200, 200)  # Gunmetal silver
    SLYTHERIN_DARK = (0, 50, 50)  # Very dark water

    # RAVENCLAW - Air Element (Blue & Bronze)
    RAVENCLAW_BLUE = (139, 69, 19)  # Deep blue (inverted for BGR)
    RAVENCLAW_MIDNIGHT = (79, 39, 0)  # Midnight slate
    RAVENCLAW_BRONZE = (140, 180, 255)  # Bronze/copper tone

    # HUFFLEPUFF - Earth Element (Yellow & Black)
    HUFFLEPUFF_YELLOW = (0, 215, 255)  # Warm yellow
    HUFFLEPUFF_EARTH = (70, 70, 70)  # Earthy brown
    HUFFLEPUFF_BLACK = (0, 0, 0)  # Deep black

    # Universal magical colors
    PATRONUS_SILVER = (220, 220, 255)  # Ethereal silver-white
    DARK_MAGIC_PURPLE = (100, 0, 100)  # Dark magic aura
    WAND_SPARK = (200, 200, 255)  # Spell spark blue
    PARCHMENT_CREAM = (230, 240, 250)  # Old parchment color
    STONE_GREY = (140, 140, 150)  # Castle stone
    GOLD_ACCENT = (0, 215, 255)  # Pure gold

    # Active theme (Gryffindor by default)
    THEME_COLOR_PRIMARY = GRYFFINDOR_SCARLET
    THEME_COLOR_SECONDARY = GRYFFINDOR_GOLD
    THEME_COLOR_ACCENT = GRYFFINDOR_AMBER

    BARRIER_THICKNESS = 4

    @staticmethod
    def get_house_theme(house_name):
        """Get color theme for different houses"""
        themes = {
            'gryffindor': {
                'primary': HarryPotterTheme.GRYFFINDOR_SCARLET,
                'secondary': HarryPotterTheme.GRYFFINDOR_GOLD,
                'accent': HarryPotterTheme.GRYFFINDOR_AMBER,
                'name': 'GRYFFINDOR'
            },
            'slytherin': {
                'primary': HarryPotterTheme.SLYTHERIN_EMERALD,
                'secondary': HarryPotterTheme.SLYTHERIN_SILVER,
                'accent': HarryPotterTheme.SLYTHERIN_DARK,
                'name': 'SLYTHERIN'
            },
            'ravenclaw': {
                'primary': HarryPotterTheme.RAVENCLAW_MIDNIGHT,
                'secondary': HarryPotterTheme.RAVENCLAW_BRONZE,
                'accent': HarryPotterTheme.RAVENCLAW_BLUE,
                'name': 'RAVENCLAW'
            },
            'hufflepuff': {
                'primary': HarryPotterTheme.HUFFLEPUFF_EARTH,
                'secondary': HarryPotterTheme.HUFFLEPUFF_YELLOW,
                'accent': (100, 180, 255),
                'name': 'HUFFLEPUFF'
            }
        }
        return themes.get(house_name.lower(), themes['gryffindor'])


class HandTracker:
    def __init__(self,
                 max_num_hands=1,
                 detection_confidence=0.6,
                 tracking_confidence=0.6):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_index_tip(self, frame, draw=True):
        """
        Returns (x, y) of index finger tip in image coordinates,
        or None if no hand detected.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        h, w, _ = frame.shape
        index_tip = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            if draw:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

            lm = hand_landmarks.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)
            index_tip = (x, y)

        return index_tip


# ============================
# Advanced Wand Magic Snake with Background
# ============================
class WandMagicSnake:
    def __init__(self, frame_width, frame_height, house='gryffindor', bg_image_path=None):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.food_radius = 16
        self.self_collision_distance = 10

        # Set house theme
        self.house = house.lower()
        self.house_theme = HarryPotterTheme.get_house_theme(self.house)
        self.theme = HarryPotterTheme()

        # Load background image
        self.background_image = None
        self.bg_opacity = 0.4
        # Opacity of background (0-1, lower = more transparent)
        self._load_background_image(bg_image_path, frame_width, frame_height)

        # Spell effects
        self.wand_sparks = []
        self.food_glow_pulse = 0
        self.magical_particles = []  # Background particles for atmosphere
        self.rune_glow = 0  # For animated UI elements

        # Atmosphere effects
        self._generate_background_particles()

        self.reset()

    def _load_background_image(self, image_path, width, height):
        """
        Load and resize background image

        Args:
            image_path: Path to the background image
            width: Target width
            height: Target height
        """
        if image_path is None:
            print("⚠️  No background image path provided. Using default background.")
            return

        if not os.path.exists(image_path):
            print(f"⚠️  Background image not found at: {image_path}")
            return

        try:
            # Read the image
            img = cv2.imread(image_path)

            if img is None:
                print(f"❌ Failed to load image: {image_path}")
                return

            # Resize image to match frame dimensions
            self.background_image = cv2.resize(img, (width, height))
            print(f"✅ Background image loaded successfully: {image_path}")
            print(f"   Resized to: {width}x{height}")

        except Exception as e:
            print(f"❌ Error loading background image: {e}")

    def _generate_background_particles(self):
        """Generate floating magical particles for background"""
        for _ in range(15):
            self.magical_particles.append({
                'x': random.uniform(0, self.frame_w),
                'y': random.uniform(0, self.frame_h),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-1, -0.2),
                'lifetime': random.randint(100, 200),
                'max_lifetime': 200,
                'size': random.uniform(1, 3)
            })

    def reset(self):
        self.points = []
        self.lengths = []
        self.current_length = 0.0
        self.target_length = 150
        self.prev_head = None

        self.food_pos = self._random_food_position()
        self.score = 0
        self.game_over = False
        self.game_over_time = None
        self.wand_sparks = []
        self.rune_glow = 0

    def _random_food_position(self):
        margin = 100
        x = random.randint(margin, self.frame_w - margin)
        y = random.randint(margin, self.frame_h - margin)
        return (x, y)

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _create_wand_spark(self, pos):
        """Create magical spark effects with Gothic aesthetic"""
        for _ in range(4):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            self.wand_sparks.append({
                'x': pos[0],
                'y': pos[1],
                'dx': dx,
                'dy': dy,
                'lifetime': 25,
                'max_lifetime': 25
            })

    def _update_magical_particles(self):
        """Update background atmospheric particles"""
        for particle in self.magical_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['lifetime'] -= 1

            # Respawn at top when lifetime ends
            if particle['lifetime'] <= 0:
                particle['x'] = random.uniform(0, self.frame_w)
                particle['y'] = self.frame_h
                particle['lifetime'] = 200
                particle['max_lifetime'] = 200

    def update(self, head_pos):
        """Update snake with wand position"""
        if self.game_over:
            return

        if head_pos is None:
            return

        if self.prev_head is None:
            self.prev_head = head_pos

        # Update atmospheric effects
        self._update_magical_particles()
        self.rune_glow = (self.rune_glow + 2) % 360
        self.food_glow_pulse = (self.food_glow_pulse + 1) % 60

        # Create magical spark trail
        self._create_wand_spark(head_pos)

        # Update spark positions
        self.wand_sparks = [
            s for s in self.wand_sparks
            if s['lifetime'] > 0
        ]
        for spark in self.wand_sparks:
            spark['x'] += spark['dx']
            spark['y'] += spark['dy']
            spark['lifetime'] -= 1

        # Add new head
        self.points.append(head_pos)
        segment_len = self._distance(head_pos, self.prev_head)
        self.lengths.append(segment_len)
        self.current_length += segment_len
        self.prev_head = head_pos

        # Trim tail
        while self.current_length > self.target_length and len(self.lengths) > 0:
            self.current_length -= self.lengths.pop(0)
            self.points.pop(0)

        # Check food collision
        if self._distance(head_pos, self.food_pos) < self.food_radius + 12:
            self.score += 1
            self.target_length += 45
            self.food_pos = self._random_food_position()

        # Check self collision
        if len(self.points) > 20:
            head = np.array(head_pos)
            body = np.array(self.points[:-10])
            if len(body) > 0:
                dists = np.linalg.norm(body - head, axis=1)
                if np.min(dists) < self.self_collision_distance:
                    self.game_over = True
                    self.game_over_time = time.time()

        # Check boundary
        x, y = head_pos
        if x <= 20 or x >= self.frame_w - 20 or y <= 20 or y >= self.frame_h - 20:
            self.game_over = True
            self.game_over_time = time.time()

    def _draw_magical_border(self, frame):
        """Draw protective magical barrier with Gothic runes"""
        h, w, _ = frame.shape
        thickness = self.theme.BARRIER_THICKNESS

        # Main border
        cv2.rectangle(
            frame,
            (thickness, thickness),
            (w - thickness, h - thickness),
            self.house_theme['primary'],
            thickness,
        )

        # Double border for Gothic effect
        cv2.rectangle(
            frame,
            (thickness + 3, thickness + 3),
            (w - thickness - 3, h - thickness - 3),
            self.house_theme['secondary'],
            1,
        )

        # Corner runes (magical symbols)
        rune_size = 15
        corners = [
            (rune_size + 5, rune_size + 5),
            (w - rune_size - 5, rune_size + 5),
            (rune_size + 5, h - rune_size - 5),
            (w - rune_size - 5, h - rune_size - 5)
        ]

        for corner in corners:
            cv2.circle(frame, corner, rune_size, self.house_theme['accent'], 2)
            cv2.circle(frame, corner, rune_size - 5, self.house_theme['accent'], 1)

    def _draw_atmospheric_background(self, frame):
        """Draw floating magical particles for atmosphere"""
        for particle in self.magical_particles:
            alpha = particle['lifetime'] / particle['max_lifetime']
            brightness = int(150 * alpha)
            color = (brightness + 50, brightness + 50, brightness + 100)
            size = int(particle['size'] * alpha)

            if size > 0:
                cv2.circle(
                    frame,
                    (int(particle['x']), int(particle['y'])),
                    size,
                    color,
                    -1
                )

    def _draw_snake_body(self, frame):
        """Draw snake with advanced gradient and glow effects"""
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                # Create gradient effect from head to tail
                intensity = int(255 * (i / max(len(self.points), 1)))

                # Blend house colors with intensity
                r = int((self.house_theme['primary'][2] * (1 - intensity / 255)) +
                        (self.house_theme['secondary'][2] * (intensity / 255)))
                g = int((self.house_theme['primary'][1] * (1 - intensity / 255)) +
                        (self.house_theme['secondary'][1] * (intensity / 255)))
                b = int((self.house_theme['primary'][0] * (1 - intensity / 255)) +
                        (self.house_theme['secondary'][0] * (intensity / 255)))

                color = (b, g, r)

                # Main body line
                cv2.line(
                    frame,
                    self.points[i - 1],
                    self.points[i],
                    color,
                    14,
                    cv2.LINE_AA,
                )

                # Glow effect (outer line)
                if i % 3 == 0:  # Every 3rd segment
                    cv2.line(
                        frame,
                        self.points[i - 1],
                        self.points[i],
                        self.house_theme['secondary'],
                        6,
                        cv2.LINE_AA,
                    )

    def _draw_snake_head(self, frame):
        """Draw Patronus head with magical aura"""
        if self.points:
            head_pos = self.points[-1]

            # Outer aura (expanding glow)
            glow_size = 20 + int(5 * math.sin(self.rune_glow * math.pi / 180))
            cv2.circle(frame, head_pos, glow_size, self.house_theme['accent'], 1)

            # Middle glow ring
            cv2.circle(frame, head_pos, 18, self.house_theme['secondary'], 2)

            # Head core
            cv2.circle(frame, head_pos, 13, self.theme.PATRONUS_SILVER, -1)

            # Head highlights (magical essence)
            cv2.circle(frame,
                       (head_pos[0] - 5, head_pos[1] - 5),
                       3, (255, 255, 255), -1)

    def _draw_snitch_food(self, frame):
        """Draw Golden Snitch with wing animation"""
        glow_radius = self.food_radius + int(6 * math.sin(self.food_glow_pulse / 15))

        # Outer glow
        cv2.circle(frame, self.food_pos, glow_radius + 2, self.house_theme['accent'], 1)

        # Main glow halo
        cv2.circle(frame, self.food_pos, glow_radius, self.house_theme['secondary'], 2)

        # Snitch body (golden sphere)
        cv2.circle(frame, self.food_pos, self.food_radius, (0, 215, 255), -1)

        # Snitch highlights (3D effect)
        cv2.circle(frame,
                   (self.food_pos[0] - 5, self.food_pos[1] - 5),
                   4, (255, 255, 200), -1)

        # Animated wings (oscillating lines)
        wing_offset = int(8 * math.sin(self.food_glow_pulse / 10))

        # Left wing
        cv2.line(frame,
                 (self.food_pos[0] - self.food_radius - 5, self.food_pos[1]),
                 (self.food_pos[0] - self.food_radius - 15 - wing_offset, self.food_pos[1] - 5),
                 self.house_theme['secondary'], 2)

        # Right wing
        cv2.line(frame,
                 (self.food_pos[0] + self.food_radius + 5, self.food_pos[1]),
                 (self.food_pos[0] + self.food_radius + 15 + wing_offset, self.food_pos[1] - 5),
                 self.house_theme['secondary'], 2)

    def _draw_wand_effects(self, frame):
        """Draw wand spark trail with fade effect"""
        for spark in self.wand_sparks:
            alpha = spark['lifetime'] / spark['max_lifetime']
            brightness = int(255 * alpha * 0.8)
            color = (brightness + 100, brightness + 100, 255)
            size = max(1, int(3 * alpha))

            cv2.circle(frame, (int(spark['x']), int(spark['y'])),
                       size, color, -1)

            # Spark glow
            if size > 1:
                cv2.circle(frame, (int(spark['x']), int(spark['y'])),
                           size + 1, self.house_theme['secondary'], 1)

    def draw(self, frame):
        """Draw complete magical Harry Potter themed game"""
        h, w, _ = frame.shape

        # Apply background image if loaded
        if self.background_image is not None:
            # Blend background with frame using opacity
            frame = cv2.addWeighted(self.background_image, self.bg_opacity, frame, 1 - self.bg_opacity, 0)

        # Background atmospheric effect
        self._draw_atmospheric_background(frame)

        # Magical border with runes
        self._draw_magical_border(frame)

        # Snake components
        self._draw_snake_body(frame)
        self._draw_snake_head(frame)

        # Golden Snitch
        self._draw_snitch_food(frame)

        # Wand effects
        self._draw_wand_effects(frame)

        # UI
        self._draw_ui(frame)

        return frame

    def _draw_ui(self, frame):
        """Draw UI with Gothic medieval parchment aesthetic"""
        h, w, _ = frame.shape

        # Top parchment box for title
        parchment_y = 10
        parchment_height = 50

        cv2.rectangle(frame, (5, parchment_y), (350, parchment_y + parchment_height),
                      (240, 240, 220), -1)  # Parchment background
        cv2.rectangle(frame, (5, parchment_y), (350, parchment_y + parchment_height),
                      self.house_theme['primary'], 3)  # House color border

        # Gothic title with shadow effect
        title_text = f"⚡ {self.house_theme['name']} MAGIC ⚡"
        cv2.putText(frame, title_text, (20, parchment_y + 35),
                    cv2.FONT_HERSHEY_COMPLEX, 0.9,
                    self.house_theme['primary'], 3)

        # Score parchment box
        score_y = parchment_y + parchment_height + 10
        score_height = 60

        cv2.rectangle(frame, (5, score_y), (350, score_y + score_height),
                      (240, 240, 220), -1)
        cv2.rectangle(frame, (5, score_y), (350, score_y + score_height),
                      self.house_theme['secondary'], 3)

        cv2.putText(frame, "Snitches Caught:", (20, score_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    self.house_theme['primary'], 2)

        cv2.putText(frame, str(self.score), (240, score_y + 35),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2,
                    self.house_theme['secondary'], 3)

        # Game over screen - Dark magic overlay
        if self.game_over:
            # Dark vignette with purple tint
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h),
                          self.theme.DARK_MAGIC_PURPLE, -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Game Over ornate box (Gothic style)
            game_over_box_w = 550
            game_over_box_h = 200
            box_x = (w - game_over_box_w) // 2
            box_y = (h - game_over_box_h) // 2

            # Outer box with shadow
            cv2.rectangle(frame, (box_x - 3, box_y - 3),
                          (box_x + game_over_box_w + 3, box_y + game_over_box_h + 3),
                          (0, 0, 0), 3)

            # Main box with parchment color
            cv2.rectangle(frame, (box_x, box_y),
                          (box_x + game_over_box_w, box_y + game_over_box_h),
                          (240, 240, 220), -1)

            # Decorative border
            cv2.rectangle(frame, (box_x, box_y),
                          (box_x + game_over_box_w, box_y + game_over_box_h),
                          self.house_theme['primary'], 4)

            # Corner decorations
            corner_size = 12
            corners = [
                (box_x + corner_size, box_y + corner_size),
                (box_x + game_over_box_w - corner_size, box_y + corner_size),
                (box_x + corner_size, box_y + game_over_box_h - corner_size),
                (box_x + game_over_box_w - corner_size, box_y + game_over_box_h - corner_size)
            ]
            for corner in corners:
                cv2.circle(frame, corner, 5, self.house_theme['accent'], -1)

            # Game Over text with dramatic styling
            cv2.putText(frame, "Game over for you ",
                        (box_x + 100, box_y + 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1.3,
                        self.house_theme['primary'], 4)

            cv2.putText(frame, f"Final Score: {self.score}",
                        (box_x + 140, box_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        self.house_theme['secondary'], 3)

            cv2.putText(frame, "R: Restart  |  Q: Quit",
                        (box_x + 130, box_y + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 2)


# ============================
# Main Loop
# ============================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
    cap.set(cv2.CAP_PROP_FPS, 30)

    success, frame = cap.read()
    if not success:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    frame_h, frame_w, _ = frame.shape
   # house_choice = 'gryffindor'  # Red & Gold
  #  house_choice = 'slytherin'  # Green & Silver
   # house_choice = 'ravenclaw'  # Blue & Bronze
   # house_choice = 'hufflepuff'  # Yellow & Earth

    # Choose house
    house_choice = 'ravenclaw'

    # PATH TO YOUR BACKGROUND IMAGE
    # Change this to your image file path
    bg_image_path = 'hogwards.jpg'  # Put your image in same folder as script
    # OR use full path: bg_image_path = r'C:\path\to\harry_potter.jpg'

    hand_tracker = HandTracker()
    game = WandMagicSnake(frame_w, frame_h, house=house_choice, bg_image_path=bg_image_path)

    prev_time = time.time()

    print("=" * 60)
    print("   ⚡ WELCOME TO WAND MAGIC SNAKE ⚡".center(60))
    print("   Harry Potter Edition with Background".center(60))
    print("=" * 60)
    print(f"House: {game.house_theme['name']}")
    print(f"Background: {bg_image_path}")
    print(f"Background Opacity: {game.bg_opacity} (adjust in code)")
    print("\nControls:")
    print("  🪄 Move your hand to guide the Patronus")
    print("  🦅 Catch the Golden Snitches")
    print("  🔄 R: Restart game")
    print("  ❌ Q: Quit")
    print("\nAvailable Houses: gryffindor, slytherin, ravenclaw, hufflepuff")
    print("=" * 60)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Get wand tip
        index_tip = hand_tracker.get_index_tip(frame, draw=True)

        # Update game
        game.update(index_tip)

        # Draw game
        frame = game.draw(frame)

        # FPS counter with house colors
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (frame_w - 200, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            game.house_theme['secondary'],
            2,
        )

        cv2.imshow("Wand Magic Snake - Harry Potter Edition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n✨ Thanks for playing! Expelliarmus! ✨")
            break
        if key == ord("r"):
            print("🔄 Restarting game...")
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
