"""Create a captioned MP4 demonstration of the verified project run."""

from __future__ import annotations

import asyncio
import subprocess
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "credit_scoring_demo.mp4"
CAPTURE = ROOT / "video_app_capture.png"
WIDTH, HEIGHT, FPS = 1280, 720, 30


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def wrapped(draw, text, xy, width, size=34, fill="#E5E7EB", spacing=12):
    words, lines, line = text.split(), [], ""
    fnt = font(size)
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    draw.multiline_text(xy, "\n".join(lines), font=fnt, fill=fill, spacing=spacing)


def slide(title, body, accent="#60A5FA", image_path=None):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "#08111F")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((48, 42, 1232, 678), 30, fill="#111C2E", outline="#263852", width=2)
    draw.rectangle((48, 42, 62, 678), fill=accent)
    draw.text((96, 80), title, font=font(48, True), fill="#F8FAFC")
    if image_path:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1120, 500))
        x, y = (WIDTH - img.width) // 2, 150
        canvas.paste(img, (x, y))
    else:
        wrapped(draw, body, (100, 180), 1060, 34)
    draw.text((100, 630), "CodeAlpha • Credit Scoring Model", font=font(22), fill="#94A3B8")
    return np.array(canvas)[:, :, ::-1]


def add_clip(writer, frame, seconds):
    for _ in range(int(seconds * FPS)):
        writer.write(frame)


async def capture_app():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 1100})
        await page.goto("http://localhost:8501", wait_until="networkidle")
        await page.get_by_role("button", name="Assess credit risk").wait_for(
            state="visible", timeout=60000
        )
        await page.get_by_role("button", name="Assess credit risk").click()
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(CAPTURE), full_page=True)
        await browser.close()


def main():
    server = subprocess.Popen(
        [
            "py",
            "-3",
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.port=8501",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen("http://localhost:8501", timeout=1).status == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Streamlit did not start.")
        asyncio.run(capture_app())
    finally:
        server.terminate()
        server.wait(timeout=10)
    writer = cv2.VideoWriter(
        str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    clips = [
        slide(
            "Credit Scoring Model",
            "CodeAlpha Machine Learning Internship\n\n"
            "German Credit classification and Streamlit deployment",
            "#22C55E",
        ),
        slide(
            "1. Project Run",
            "Command:\npy -3 train_model.py\n\n"
            "Dataset: 1,000 rows and 20 original features\n"
            "Stratified train/test split: 800 / 200\n"
            "Training completed successfully.",
        ),
        slide(
            "2. Models Compared",
            "Logistic Regression\nDecision Tree\nRandom Forest\n\n"
            "Evaluation metrics:\nAccuracy • Precision • Recall • F1-Score • ROC-AUC",
            "#A78BFA",
        ),
        slide(
            "3. Verified Results",
            "Best model: Random Forest\n\n"
            "Accuracy: 0.775\nPrecision: 0.647\nRecall: 0.550\n"
            "F1-Score: 0.595\nROC-AUC: 0.808\n\nAutomated tests: 3 passed",
            "#F59E0B",
        ),
        slide("4. ROC Curve", "", "#38BDF8", ROOT / "images" / "roc_curve.png"),
        slide(
            "5. Confusion Matrix",
            "",
            "#38BDF8",
            ROOT / "images" / "confusion_matrix.png",
        ),
        slide(
            "6. Live Streamlit Prediction",
            "",
            "#22C55E",
            CAPTURE,
        ),
        slide(
            "Project Completed",
            "Complete source code includes:\n\n"
            "Training pipeline • Jupyter notebook • Saved model\n"
            "Evaluation images • Streamlit application • Automated tests\n\n"
            "GitHub repository: CodeAlpha_CreditScoringModel",
            "#22C55E",
        ),
    ]
    durations = [4, 6, 6, 7, 5, 5, 7, 5]
    for frame, seconds in zip(clips, durations):
        add_clip(writer, frame, seconds)
    writer.release()
    print(f"Created: {OUT}")
    print(f"Duration: {sum(durations)} seconds")
    print(f"Resolution: {WIDTH}x{HEIGHT}, {FPS} FPS")


if __name__ == "__main__":
    main()
