import os
import base64
import numpy as np
import cv2
import pytesseract
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set Windows local Tesseract binary path if running directly on Windows
if os.name == "nt":
    default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_win_path):
        pytesseract.pytesseract.tesseract_cmd = default_win_path

app = FastAPI(title="VTOP Captcha Solver API")

# Enable Cross-Origin Resource Sharing for Chrome extension calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptchaRequest(BaseModel):
    base64_image: str

def extract_text(image_np: np.ndarray, psm_mode: int) -> str:
    """Helper to run OCR with full alphanumeric matching and forced uppercase conversion."""
    config = (
        f"--oem 3 --psm {psm_mode} "
        r"-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    raw = pytesseract.image_to_string(image_np, config=config)
    cleaned = "".join(raw.split()).strip().upper()
    return cleaned

@app.get("/")
def health_check():
    return {"status": "online", "service": "VTOP Captcha Solver"}

@app.post("/solve")
def solve_captcha(payload: CaptchaRequest):
    try:
        raw_b64 = payload.base64_image
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]

        image_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image data.")

        # 1. Upscale image 3x to expand letter separation and line thickness
        scaled = cv2.resize(img, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # 3. Bilateral filter preserves character boundaries while removing noise
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # 4. Otsu's binary thresholding (dark text on white canvas)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Morphological erosion to break thin connections between touching letters
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.erode(thresh, kernel, iterations=1)

        # Pass 1: Try single-line recognition (PSM 7)
        result = extract_text(processed, psm_mode=7)

        # Pass 2: Fallback to single-word (PSM 8) or un-eroded threshold if empty or malformed
        if len(result) < 4:
            alt_result = extract_text(thresh, psm_mode=8)
            if len(alt_result) > len(result):
                result = alt_result

        # Truncate to standard VTOP 6-character length if stray symbols were captured
        if len(result) > 6:
            result = result[:6]

        return {"status": "success", "captcha": result}

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
