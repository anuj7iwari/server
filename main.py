from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
import pytesseract

app = FastAPI()

# IMPORTANT: Allow your Chrome Extension to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you can restrict this to the VTOP domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CaptchaRequest(BaseModel):
    base64_image: str

@app.post("/solve")
def solve_captcha(request: CaptchaRequest):
    try:
        # 1. Clean the base64 string (remove the "data:image/jpeg;base64," part if sent by the extension)
        b64_string = request.base64_image
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        # 2. Decode base64 to image bytes
        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data))

        # 3. Pre-process the image for better OCR accuracy (Grayscale & Thresholding)
        image = image.convert('L') # Convert to grayscale
        # Apply a threshold to make the text bold black and background white
        threshold = 150
        image = image.point(lambda p: 255 if p > threshold else 0)

        # 4. Run OCR (VTOP captchas are usually 6 characters, uppercase and numbers)
        # psm 8 assumes a single word/line. Whitelist ensures no special characters are guessed.
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Clean up whitespace
        text = text.strip().replace(" ", "")

        return {"status": "success", "captcha": text}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def home():
    return {"message": "Captcha Solver API is running!"}