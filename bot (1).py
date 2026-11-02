import asyncio
import io
import logging
import os

from nudenet import NudeDetector
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("censor-bot")

load_dotenv()  # reads variables from the .env file next to the script

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN is missing. Set it in the .env file.")

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "blur").lower()  # blur | pixelate

# Classes that get censored. Add the *_COVERED ones if you want stricter filtering.
CENSOR_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))  # detection confidence threshold
PADDING = float(os.getenv("PADDING", "0.15"))      # grow each box so edges are fully covered
BLUR_RADIUS_DIV = 6   # larger = weaker blur relative to box size

detector = NudeDetector()  # loads the ONNX model once


def censor_image(data: bytes, mode: str = "blur") -> tuple[bytes, int]:
    """Return (jpeg_bytes, number_of_regions_censored)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # NudeDetector wants a path or a numpy array; save a temp buffer via numpy
    import numpy as np

    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR (cv2 style)
    detections = detector.detect(arr)

    count = 0
    for d in detections:
        if d["class"] not in CENSOR_CLASSES or d["score"] < MIN_SCORE:
            continue
        x, y, w, h = d["box"]
        px, py = int(w * PADDING), int(h * PADDING)
        left = max(0, x - px)
        top = max(0, y - py)
        right = min(img.width, x + w + px)
        bottom = min(img.height, y + h + py)
        if right <= left or bottom <= top:
            continue

        region = img.crop((left, top, right, bottom))
        if mode == "pixelate":
            small = region.resize(
                (max(1, region.width // 16), max(1, region.height // 16)),
                Image.BILINEAR,
            )
            region = small.resize(region.size, Image.NEAREST)
        else:
            radius = max(8, max(region.size) // BLUR_RADIUS_DIV)
            region = region.filter(ImageFilter.GaussianBlur(radius))
        img.paste(region, (left, top))
        count += 1

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue(), count


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a photo and I'll blur sensitive/private areas.\n\n"
        "Commands:\n"
        "/mode blur - Gaussian blur (default)\n"
        "/mode pixelate - mosaic effect\n\n"
        "Tip: send as a file (document) to keep full quality."
    )


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = (context.args[0].lower() if context.args else "")
    if choice not in ("blur", "pixelate"):
        await update.message.reply_text("Usage: /mode blur  or  /mode pixelate")
        return
    context.user_data["mode"] = choice
    await update.message.reply_text(f"Mode set to {choice}.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    as_document = False

    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
        as_document = True
    else:
        return

    status = await msg.reply_text("Processing...")
    try:
        data = bytes(await tg_file.download_as_bytearray())
        mode = context.user_data.get("mode", DEFAULT_MODE)
        loop = asyncio.get_running_loop()
        result, count = await loop.run_in_executor(None, censor_image, data, mode)

        caption = (
            f"Censored {count} region(s)." if count else "No sensitive areas detected."
        )
        bio = io.BytesIO(result)
        if as_document:
            bio.name = "censored.jpg"
            await msg.reply_document(bio, caption=caption)
        else:
            await msg.reply_photo(bio, caption=caption)
    except Exception:
        log.exception("processing failed")
        await msg.reply_text("Sorry, I couldn't process that image.")
    finally:
        await status.delete()


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mode", set_mode))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_image))
    log.info("Bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
