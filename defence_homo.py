
from functools import partial
from text_OCR import text_OCR_text
from textattack.utils import text_homo_back

defence_method={
    'ocr':text_OCR_text,
    'del':partial(
        text_homo_back,
        style='del'
    ),
    'map':partial(
        text_homo_back,
        style='map'
    )
}