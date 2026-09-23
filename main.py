from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import pytesseract
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptchaRequest(BaseModel):
    base64_image: str

@app.post("/solve")
def solve_captcha(request: CaptchaRequest):
    try:
        b64_string = request.base64_image
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # 1. Convert base64 straight into an OpenCV image (numpy array)
        image_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid image data")

        # 2. Resize image (make it 2x larger) - Tesseract loves big text
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 3. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. Apply a Median Blur to remove background dots/noise
        blur = cv2.medianBlur(gray, 3)

        # 5. Apply Otsu's Thresholding (Automatically finds the best black/white contrast)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Run OCR
        # psm 8 = single word. Whitelist limits guesses to only valid VTOP characters.
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        # Clean up the final text (VTOP captchas are 6 characters)
        text = text.strip().replace(" ", "")
        
        # Optional: Force it to exactly 6 characters if it guessed extra noise
        if len(text) > 6:
            text = text[:6]

        return {"status": "success", "captcha": text}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Advanced Captcha Solver API is running!"}
