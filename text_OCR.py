from PIL import Image, ImageDraw, ImageFont
import pytesseract

def text_OCR_text(text, img_path=None):
    # Step 2: Print the text into an image.
    # Load a default font
    font = ImageFont.load_default()
    
    # Create a temporary image to calculate text size.
    temp_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_img)
    # textbbox returns (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Define some padding around the text.
    padding = 10
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # Create the final image with a white background.
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), text, fill='black', font=font)

    if img_path is not None:
        img.save(img_path)

    # Step 3: Read text from the image using OCR.
    # If Tesseract is not in your PATH, specify its full path like:
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text
    # print("Extracted text from image:")
    # print(extracted_text)
    # rel=[1 for t in text if t=='B']
    # print(sum(rel)/len(text))

if __name__ == "__main__":
    text = "Step 1: Input a text from the user ʙΒВвᏴᏼᗷᛒℬꓐꞴＢ𐊂𐊡𐌁𝐁𝐵𝑩𝓑𝔅𝔹𝕭𝖡𝗕𝘉𝘽𝙱𝚩𝛣𝜝𝝗𝞑"
    text_OCR_text(text)
