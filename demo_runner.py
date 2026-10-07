"""
Demo Media Synthesis and Verification Script.
Generates genuine test media for offline zero-internet demonstration:
- demo/demo_real.jpg & demo/demo_fake.jpg
- demo/demo_real.wav & demo/demo_fake.wav
- demo/demo_real.mp4 & demo/demo_fake.mp4
"""

import os
import cv2
import numpy as np
from scipy.io import wavfile

def generate_demo_images(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    h, w = 512, 512

    # 1. Real Image: Natural face geometry with natural skin pore micro-texture and continuous gradients
    real_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background soft gradient
    for y in range(h):
        real_img[y, :] = [int(40 + 20 * (y / h)), int(45 + 15 * (y / h)), int(50 + 25 * (y / h))]

    # Natural head oval
    cv2.ellipse(real_img, (w//2, h//2), (160, 210), 0, 0, 360, (180, 195, 220), -1)
    # Natural eyes
    cv2.circle(real_img, (w//2 - 60, h//2 - 40), 18, (60, 50, 40), -1)
    cv2.circle(real_img, (w//2 + 60, h//2 - 40), 18, (60, 50, 40), -1)
    # Natural mouth
    cv2.ellipse(real_img, (w//2, h//2 + 90), (45, 16), 0, 0, 360, (120, 110, 160), -1)
    # Add subtle natural camera sensor noise / skin texture
    noise = np.random.normal(0, 7.0, (h, w, 3)).astype(np.float32)
    real_img = np.clip(real_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    real_path = os.path.join(output_dir, "demo_real.jpg")
    cv2.imwrite(real_path, real_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # 2. Fake Image: Injected deepfake artifacts
    # - Over-smoothed inner face (diffusion blur / GAN lack of pore texture)
    # - Sharp Poisson blending seam boundary around the jawline
    # - High-frequency checkerboard grid anomaly in frequency domain
    fake_img = real_img.copy()
    # Blur inner face artificially
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(inner_mask, (w//2, h//2 + 20), (120, 140), 0, 0, 360, 255, -1)
    blurred_face = cv2.GaussianBlur(fake_img, (25, 25), 10)
    fake_img[inner_mask == 255] = blurred_face[inner_mask == 255]

    # Draw discontinuous blending seam border along the jaw
    cv2.ellipse(fake_img, (w//2, h//2 + 20), (122, 142), 0, 40, 140, (140, 130, 235), 3)

    # Inject periodic high-frequency checkerboard pattern (GAN upsampling artifact)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_pattern = ((grid_y % 4 == 0) & (grid_x % 4 == 0)).astype(np.float32) * 22.0
    fake_img[:, :, 0] = np.clip(fake_img[:, :, 0].astype(np.float32) + grid_pattern, 0, 255).astype(np.uint8)

    fake_path = os.path.join(output_dir, "demo_fake.jpg")
    cv2.imwrite(fake_path, fake_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[OK] Generated {real_path} and {fake_path}")

def generate_demo_audio(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Real Audio: Human speech simulation with varying pitch (f0 contour 120-180Hz) + harmonics + noise
    f0 = 140 + 35 * np.sin(2 * np.pi * 1.2 * t) + 15 * np.sin(2 * np.pi * 3.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    real_audio = (
        0.5 * np.sin(phase) +
        0.25 * np.sin(2 * phase) +
        0.12 * np.sin(3 * phase) +
        0.05 * np.random.normal(0, 0.1, len(t))
    )
    # Modulate envelope for words
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)**2
    real_audio = (real_audio * env).astype(np.float32)
    real_audio = real_audio / np.max(np.abs(real_audio))
    real_audio_i16 = (real_audio * 32767).astype(np.int16)

    real_path = os.path.join(output_dir, "demo_real.wav")
    wavfile.write(real_path, sr, real_audio_i16)

    # 2. Fake Audio: Monotonic robotic flat pitch (exactly 150 Hz with zero jitter) + sharp vocoder brickwall filter
    flat_f0 = 150.0
    phase_flat = 2 * np.pi * flat_f0 * t
    fake_audio = 0.6 * np.sin(phase_flat) + 0.3 * np.sin(2 * phase_flat)
    fake_audio = fake_audio * env
    # Brickwall cutoff: remove all frequency above 4000 Hz sharply
    from scipy import signal
    sos = signal.butter(10, 3800, 'lowpass', fs=sr, output='sos')
    fake_audio = signal.sosfilt(sos, fake_audio)
    fake_audio = fake_audio / (np.max(np.abs(fake_audio)) + 1e-6)
    fake_audio_i16 = (fake_audio * 32767).astype(np.int16)

    fake_path = os.path.join(output_dir, "demo_fake.wav")
    wavfile.write(fake_path, sr, fake_audio_i16)
    print(f"[OK] Generated {real_path} and {fake_path}")

def generate_demo_videos(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    real_vpath = os.path.join(output_dir, "demo_real.mp4")
    fake_vpath = os.path.join(output_dir, "demo_fake.mp4")

    # Generate 60-frame videos at 20 fps
    fps = 20.0
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 1. Real Video: Smooth head sway and synchronous mouth opening
    out_real = cv2.VideoWriter(real_vpath, fourcc, fps, (w, h))
    for i in range(50):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        # Sway
        dx = int(12 * np.sin(i / 8.0))
        cv2.circle(frame, (w//2 + dx, h//2), 55, (170, 190, 220), -1)
        # Mouth smooth motion
        mouth_h = int(6 + 8 * (np.sin(i / 4.0) + 1.0))
        cv2.ellipse(frame, (w//2 + dx, h//2 + 25), (18, mouth_h), 0, 0, 360, (80, 70, 110), -1)
        out_real.write(frame)
    out_real.release()

    # 2. Fake Video: Injected frame jitter and temporal flicker
    out_fake = cv2.VideoWriter(fake_vpath, fourcc, fps, (w, h))
    for i in range(50):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        dx = int(12 * np.sin(i / 8.0))
        # Sudden flickering on frames 18-32 (injected deepfake sequence)
        is_manip_frame = (18 <= i <= 32)
        if is_manip_frame:
            # Over-smoothed face with jittered position
            jitter = np.random.randint(-6, 7)
            cv2.circle(frame, (w//2 + dx + jitter, h//2), 55, (190, 210, 240), -1)
            cv2.circle(frame, (w//2 + dx + jitter, h//2 + 25), 20, (110, 100, 255), 2) # seam border
        else:
            cv2.circle(frame, (w//2 + dx, h//2), 55, (170, 190, 220), -1)
        out_fake.write(frame)
    out_fake.release()

    print(f"[OK] Generated {real_vpath} and {fake_vpath}")

def main():
    demo_dir = "data/demo"
    print(f"Generating authentic vs synthetic test media in {demo_dir}...")
    generate_demo_images(demo_dir)
    generate_demo_audio(demo_dir)
    generate_demo_videos(demo_dir)
    print("\n[SUCCESS] All zero-internet local test media generated successfully.")

if __name__ == "__main__":
    main()
