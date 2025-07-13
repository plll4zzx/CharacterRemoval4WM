import unicodedata

def normalize_text(s):
    return unicodedata.normalize('NFKC', s)#.casefold()

text_list = ['café', 'Café', 'ＣＡＦＥ']
normalized = [normalize_text(t) for t in text_list]
print(normalized) 

from confusable_homoglyphs import confusables

def normalize_char(input_str: str):
    normal_res=confusables.is_confusable(input_str, greedy=True, preferred_aliases=['latin'])
    for res in normal_res:
        input_str=input_str.replace(res['character'], res['homoglyphs'][0]['c'])
    return input_str

char_str="ΑАᎪᗅᴀꓮꭺＡ𐊠𖽀𝐀𝐴𝑨𝖠𝗔𝘈𝘼𝙰𝚨𝛢𝜜𝝖𝞐"
print(char_str)
nor_str=normalize_text(char_str)
print(nor_str)
for a in nor_str:
    print(a, a=='A')