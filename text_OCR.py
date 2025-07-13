from PIL import Image, ImageDraw, ImageFont
# import mmocr.utils
import pytesseract
import numpy as np
from paddleocr import PaddleOCR
# from mmocr.apis import MMOCR


ocr = PaddleOCR(use_angle_cls=True, lang="en") 
def text_OCR_text(text, img_path=None, style='ocr_p'):
    
    img=text_to_image(text)
    if img_path is not None:
        img.save(img_path)
    
    if style=='ocr_t':
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text
    else:
        img=np.array(img)
        try:
            extracted_text = ocr.ocr(img, cls=True)[0][0][1][0]
        except:
            return text
        return extracted_text

def find_max_font_size(text, max_width, max_height, font_path, line_spacing, padding, bg_color):
    min_size, max_size = 10, 200  # Define a reasonable font size range
    best_size = min_size
    while min_size <= max_size:
        mid_size = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, mid_size)
        # font = ImageFont.load_default()
        dummy_img = Image.new("RGB", (1, 1), bg_color)
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)
        
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding
        
        if img_width <= max_width and img_height <= max_height:
            best_size = mid_size
            min_size = mid_size + 1  # Try larger font
        else:
            max_size = mid_size - 1  # Try smaller font
    return best_size

def text_to_image(
    text, max_width=600, max_height=200, font_color="black", bg_color="white",
    # font_path="/usr/share/fonts/truetype/unifont/unifont.ttf",#
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    padding=10, line_spacing=4
):
    """
    Convert a text string into an image with the largest possible font size while minimizing whitespace.
    """
    # Function to find the best font size that fits within max_width and max_height

    # Find the best font size dynamically
    optimal_font_size = find_max_font_size(text, max_width, max_height, font_path, line_spacing, padding, bg_color)
    font = ImageFont.truetype(font_path, optimal_font_size)

    # Create dummy image to get text bounding box
    dummy_img = Image.new("RGB", (1, 1), bg_color)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=line_spacing)

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Create final image with optimized font size
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((padding, padding), text, font=font, fill=font_color, spacing=line_spacing)

    return img


import homoglyphs as hg
hgc = hg.Homoglyphs(categories=('CYRILLIC', )) #'LATIN', 'COMMON', 
homoglyphs = hg.Homoglyphs(
    languages={'en'},
    strategy=hg.STRATEGY_LOAD,
    ascii_strategy=hg.STRATEGY_REMOVE,
)
if __name__ == "__main__":
    text = "Step 1: Input a text from the user BʙΒВвᏴᏼᗷᛒℬꓐꞴＢ𐊂𐊡𐌁𝐁𝐵𝑩𝓑𝔅𝔹𝕭𝖡𝗕𝘉𝘽𝙱𝚩𝛣𝜝𝝗𝞑"
    print(text)
    t=text_OCR_text(text)
    print(t)