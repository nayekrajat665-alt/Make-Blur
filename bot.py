import asyncio
import io
import logging
import os

import numpy as np
from dotenv import load_dotenv
from nudenet import NudeDetector
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


def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


DEFAULT_MODE = os.getenv("DEFAULT_MODE", "blur").lower()  # blur | pixelate
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))         # real-photo detector threshold
ANIME_SCORE = float(os.getenv("ANIME_SCORE", "0.25"))     # anime detector threshold
PADDING = float(os.getenv("PADDING", "0.15"))             # grow boxes so edges are covered
DETECT_ANIME = env_bool("DETECT_ANIME", True)
INCLUDE_COVERED = env_bool("INCLUDE_COVERED", False)      # also blur clothed/covered areas
BLUR_RADIUS_DIV = 6                                       # larger = weaker blur

# --- Real-photo detector (NudeNet) -------------------------------------------
CENSOR_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}
if INCLUDE_COVERED:
    CENSOR_CLASSES |= {
        "FEMALE_GENITALIA_COVERED",
        "FEMALE_BREAST_COVERED",
        "BUTTOCKS_COVERED",
        "ANUS_COVERED",
    }

nude_detector = NudeDetector()

# --- Anime/cartoon detector (dghs-imgutils) ----------------------------------
detect_censors = None
if DETECT_ANIME:
    try:
        from imgutils.detect import detect_censors  # downloads the model on first use
    except Exception:
        log.exception("Anime detector unavailable; continuing with NudeNet only")
        detect_censors = None


def find_boxes(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return [(left, top, right, bottom)] for every area that should be censored."""
    boxes: list[tuple[int, int, int, int]] = []

    # Real photos
    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
    for d in nude_detector.detect(arr):
        if d["class"] in CENSOR_CLASSES and d["score"] >= MIN_SCORE:
            x, y, w, h = d["box"]
            boxes.append((int(x), int(y), int(x + w), int(y + h)))

    # Anime / drawn art
    if detect_censors is not None:
        try:
            for (x0, y0, x1, y1), _label, score in detect_censors(
                img, conf_threshold=ANIME_SCORE
            ):
                boxes.append((int(x0), int(y0), int(x1), int(y1)))
        except Exception:
            log.exception("anime detection failed on this image")

    return boxes


def censor_image(data: bytes, mode: str = "blur") -> tuple[bytes, int]:
    """Return (jpeg_bytes, number_of_regions_censored)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    boxes = find_boxes(img)

    count = 0
    for x0, y0, x1, y1 in boxes:
        pw, ph = int((x1 - x0) * PADDING), int((y1 - y0) * PADDING)
        left = max(0, x0 - pw)
        top = max(0, y0 - ph)
        right = min(img.width, x1 + pw)
        bottom = min(img.height, y1 + ph)
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
        "Send me a photo or anime/cartoon image and I'll blur sensitive areas.\n\n"
        "Commands:\n"
        "/mode blur - Gaussian blur (default)\n"
        "/mode pixelate - mosaic effect\n\n"
        "Tip: send as a file (document) to keep full quality."
    )


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = context.args[0].lower() if context.args else ""
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
    log.info("Bot running (anime detector: %s)", "on" if detect_censors else "off")
    app.run_polling()


if __name__ == "__main__":
    main()
