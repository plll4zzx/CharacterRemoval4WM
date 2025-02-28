from PIL import Image, ImageDraw, ImageFont
import pytesseract
# from mmocr.utils.ocr import MMOCR

def create_high_quality_image(text, scale_factor=2):
    # 使用高质量的 TrueType 字体，请确保字体文件路径正确
    # font_path = "arial.ttf"  # 请替换为系统中存在的 TTF 字体文件路径
    # base_font_size = 20
    # 放大字号以适应高分辨率绘制
    # font = ImageFont.truetype(font_path, base_font_size * scale_factor)
    
    font = ImageFont.load_default()
    
    # 创建临时图像计算文字尺寸（使用 textbbox）
    temp_img = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 根据放大倍数调整边距
    padding = 10 * scale_factor
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding

    # 创建高分辨率图像
    high_res_img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(high_res_img)
    draw.text((padding, padding), text, fill='black', font=font)

    # 缩小图像至目标尺寸，使用抗锯齿以获得更清晰的效果
    final_img = high_res_img.resize((img_width // scale_factor, img_height // scale_factor), Image.Resampling.LANCZOS) 
    return final_img

def text_OCR_text(text, img_path=None):
    # Step 2: Print the text into an image.
    # Load a default font
    # font = ImageFont.load_default()
    
    # # Create a temporary image to calculate text size.
    # temp_img = Image.new('RGB', (1, 1))
    # draw = ImageDraw.Draw(temp_img)
    # # textbbox returns (left, top, right, bottom)
    # bbox = draw.textbbox((0, 0), text, font=font)
    # text_width = bbox[2] - bbox[0]
    # text_height = bbox[3] - bbox[1]

    # # Define some padding around the text.
    # padding = 10
    # img_width = text_width + 2 * padding
    # img_height = text_height + 2 * padding

    # # Create the final image with a white background.
    # img = Image.new('RGB', (img_width, img_height), color='white')
    # draw = ImageDraw.Draw(img)
    # draw.text((padding, padding), text, fill='black', font=font)
    img=text_to_image(text)
    # img=create_high_quality_image(text)
    if img_path is not None:
        img.save(img_path)
    # Step 3: Read text from the image using OCR.
    # If Tesseract is not in your PATH, specify its full path like:
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    extracted_text = pytesseract.image_to_string(img)
    # ocr = MMOCR(det="DBNet", recog="CRNN")  # 文本检测 + 识别
    # extracted_text = ocr.readtext(img, output="output.jpg")
    return extracted_text
    # print("Extracted text from image:")
    # print(extracted_text)
    # rel=[1 for t in text if t=='B']
    # print(sum(rel)/len(text))

def text_to_image(text, max_width=600, max_height=200, font_color="black", bg_color="white",
                #   font_path="/usr/share/fonts/truetype/unifont/unifont.ttf",#
                  font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  padding=10, line_spacing=4):
    """
    Convert a text string into an image with the largest possible font size while minimizing whitespace.
    """
    # Function to find the best font size that fits within max_width and max_height
    def find_max_font_size(text, max_width, max_height, font_path, line_spacing, padding):
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

    # Find the best font size dynamically
    optimal_font_size = find_max_font_size(text, max_width, max_height, font_path, line_spacing, padding)
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

if __name__ == "__main__":
    # text = "Step 1: Input a text from the user BʙΒВвᏴᏼᗷᛒℬꓐꞴＢ𐊂𐊡𐌁𝐁𝐵𝑩𝓑𝔅𝔹𝕭𝖡𝗕𝘉𝘽𝙱𝚩𝛣𝜝𝝗𝞑"
    # print(text)
    from textattack.utils import homos
    for key in homos:
        homo=homos[key]
        # print(homo, len(homo))
        # for h in homo:
        #     ocr_char=text_OCR_text(h, img_path='text.png')
        #     if ocr_char==key or ocr_char==h:
        #         print(key, h, ocr_char)
        ocr_str=text_OCR_text(homo, img_path='plot/text'+key+'.png')
        # print("'{key}':".format(key=key), end='')
        # print(ocr_str.replace('\n',' '), len(ocr_str))
        # for id,ocr_char in enumerate(ocr_str):
        #     lo=homo.find(ocr_char)
        #     if ocr_char==key or lo>-1:
        #         continue
        #         # print(key, homo[lo], ocr_char)
        #     else:
        #         print(id,',')
        #         break
    # print()
