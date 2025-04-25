
from functools import partial
from text_OCR import text_OCR_text
from textattack.utils import text_homo_back
from spellcheck import spell_check_ltp,spell_check_TextBlob

defence_method={
    'ocr_t':partial(
        text_OCR_text,
        style='ocr_t'
    ),
    'ocr':partial(
        text_OCR_text,
        style='ocr_t'
    ),
    'del':partial(
        text_homo_back,
        style='del'
    ),
    'map':partial(
        text_homo_back,
        style='map'
    ),
    'spell_check_ltp':partial(
        spell_check_ltp,
    ),
    'spell_check_TextBlob':partial(
        spell_check_TextBlob,
    )
}