import json
import logging
from logging import handlers
import random

unprintable_char=''.join([chr(i) for i in range(1000) if chr(i).isprintable()==False])[0:10]
homos = {
    "A":"ΑАᎪᗅᴀꓮꭺＡ𐊠𖽀𝐀𝐴𝑨𝖠𝗔𝘈𝘼𝙰𝚨𝛢𝜜𝝖𝞐",
    'B':'ʙΒВвᏴᏼᗷꓐＢ𐊂𐊡𐌁𝐁𝐵𝑩𝖡𝗕𝘉𝘽𝙱𝚩𝛣𝜝𝝗𝞑',
    'C':'ϹСᏟℂⅭ⊂Ⲥ⸦ꓚＣ𐐕𝐂𝐶𝑪𝖢𝗖𝘊𝘾𝙲🝌',
    'D':'ᎠᗞᗪᴅⅅⅮꓓꭰＤ𝐃𝐷𝑫𝔻𝖣𝗗𝘋𝘿𝙳',
    'E':'ΕЕᎬᴇ⋿ⴹꓰꭼＥ𐊆𑢦𑢮𝐸𝔼𝖤𝘌𝙀𝙴𝚬𝛦𝜠𝝚𝞔',
    'F':'ϜᖴꓝＦ𐊇𐊥𐔥𑢢𝈓𝐅𝐹𝑭𝖥𝗙𝘍𝙁𝙵𝟊',
    'G':'ɢԌԍᏀꓖꮐＧ𝐆𝐺𝑮𝖦𝗚𝘎𝙂𝙶',
    'H':'ʜΗНнᎻᕼⲎꓧꮋＨ𐋏𝐇𝐻𝑯𝖧𝗛𝘏𝙃𝙷𝚮𝛨𝜢𝝜𝞖',
    'I':'ɪΙІӀӏⅠⲒＩ𝐈𝐥𝐼𝑰𝕀𝗹𝙡𝙸𝚕𝚰𝛪𝜤',
    'J':'ͿЈᎫᒍᴊꓙꞲꭻＪ𝐉𝐽𝑱𝒥𝖩𝗝𝘑𝙅𝙹',
    'K':'ΚКᏦᛕKⲔꓗＫ𐔘𝐊𝐾𝑲𝖪𝗞𝘒𝙆𝙺𝚱𝛫𝜥𝝟𝞙',
    'L':'ʟᏞᒪⳐⳑꓡꮮＬ𐐛𐑃𐔦𑢣𑢲𖼖𝈪𝐋𝐿𝑳𝓛𝕃𝖫𝗟𝘓𝙇𝙻',
    'M':'ΜϺМᎷᛖⅯⲘꓟＭ𐊰𐌑𝐌𝑀𝑴𝖬𝗠𝘔𝙈𝙼𝚳𝛭𝜧𝝡𝞛',
    'N':'ɴΝℕⲚꓠＮ𐔓𝐍𝑁𝑵𝖭𝗡𝘕𝙉𝙽𝚴𝛮𝜨𝝢𝞜',
    'O':'𐊒𐊫𐐄𐓂𑢵𑣠𝐎𝑂𝑶𝒪𝓞𝖮𝗢𝘖𝙊𝙾𝚶𝛰𝜪𝝤𝞞',
    'P':'ΡРᏢᑭᴘᴩℙⲢꓑꮲＰ𐊕𝐏𝑃𝑷𝖯𝗣𝘗𝙋𝙿𝚸𝛲𝜬𝝦𝞠',
    'Q':'ℚⵕＱ𝐐𝑄𝑸𝒬𝓠𝖰𝗤𝘘𝙌𝚀',
    'R':'ƦʀᎡᏒᖇᚱℝꓣꭱꮢＲ𐒴𖼵𝈖𝐑𝑅𝑹𝓡𝕽𝖱𝗥𝘙𝙍𝚁',
    'S':'ЅՏᏕᏚꓢＳ𐊖𐐠𖼺𝐒𝑆𝑺𝒮𝓢𝔖𝕊𝕾𝖲𝗦𝘚𝙎𝚂',
    'T':'ΤТтᎢᴛ⊤⟙ⲦꓔꭲＴ𐊗𐊱𐌕𑢼𖼊𝐓𝑇𝑻𝖳𝗧𝘛𝙏𝚃𝚻𝛵𝜯𝝩𝞣🝨',
    'U':'Սሀᑌ∪⋃ꓴＵ𐓎𑢸𖽂𝐔𝑈𝑼𝒰𝓤𝖴𝗨𝘜𝙐𝚄',
    'V':'Ѵ٧۷ᏙᐯⅤⴸꓦꛟＶ𐔝𑢠𖼈𝈍𝐕𝑉𝑽𝖵𝗩𝘝𝙑𝚅',
    'W':'ԜᎳᏔꓪＷ𑣦𝐖𝑊𝑾𝒲𝖶𝗪𝘞𝙒𝚆',
    'X':'ΧХ᙭ᚷⅩⲬⵝꓫＸ𐊐𐊴𐌗𐌢𐔧𑣬𝐗𝑋𝑿𝖷𝗫𝘟𝙓𝚇𝚾𝛸𝜲𝝬𝞦',
    'Y':'ΥϒУҮᎩⲨꓬＹ𐊲𑢤𖽃𝐘𝑌𝒀𝖸𝗬𝘠𝙔𝚈𝚼𝛶𝜰𝝪𝞤',
    'Z':'ΖᏃℤꓜＺ𐋵𑢩𝐙𝑍𝒁𝖹𝗭𝘡𝙕𝚉𝚭𝛧𝜡𝝛𝞕',
    "a": "ɑɑαа⍺ａ𝐚𝑎𝒂𝒶𝓪𝔞𝖺𝗮𝘢𝙖𝚊𝛂𝛼𝜶𝝰𝞪",
    "b": "ЬƄЬᏏᑲᖯｂ𝐛𝑏𝒃𝔟𝖻𝗯𝘣𝙗𝚋",
    "c": "ϲϲсᴄⅽⲥꮯｃ𐐽𝐜𝑐𝒄𝒸𝓬𝔠𝕔𝖈𝖼𝗰𝘤𝙘𝚌",
    "d": "ԁԁᏧᑯⅆⅾꓒｄ𝐝𝑑𝒅𝒹𝓭𝖽𝗱𝘥𝙙𝚍",
    "e": "ееҽ℮ℯⅇꬲｅ𝐞𝑒𝒆𝓮𝔢𝕖𝖊𝖾𝗲𝘦𝙚𝚎",
    "f": "𝚏ϝẝꞙꬵｆ𝐟𝑓𝒇𝔣𝕗𝖋𝖿𝗳𝘧𝙛𝚏𝟋",
    "g": "ɡɡցｇ𝐠𝑔𝒈𝓰𝔤𝖌𝗀𝗴𝘨𝙜𝚐",
    "h": "հһհᏂℎｈ𝐡𝒉𝕙𝗁𝗵𝘩𝙝𝚑",
    "i": "іiıｉ𝐢𝑖𝒊𝒾𝓲𝔦𝖎𝗂𝗶𝘪𝙞𝚒",
    "j": "ϳϳјⅉｊ𝐣𝑗𝒋𝒿𝓳𝔧𝕛𝖏𝗃𝗷𝘫𝙟𝚓",
    "k": "𝚔𝒌ｋ𝐤𝑘𝒌𝗄𝗸𝘬𝙠",
    "l": "ⅼ|ƖᛁℓⅼⲒⵏꓲ𝚕𝝞𝞘",
    "m": "ｍⅿ",
    "n": "ոոռｎ𝐧𝑛𝒏𝓃𝓷𝔫𝕟𝖓𝗇𝗻𝘯𝙣𝚗",
    "o": "оοо𐐬𐓪𐔖𑓐𝐨𝑜𝒐𝖔𝗈𝗼𝘰𝙤𝚘𝛐𝜊𝝄𝝾𝞸",
    "p": "рρр⍴ⲣｐ𝐩𝑝𝒑𝖕𝗉𝗽𝘱𝙥𝚙𝛒𝝆𝞀𝞺",
    "q": "ԛԛգզｑ𝐪𝑞𝒒𝓆𝓺𝖖𝗊𝗾𝘲𝙦𝚚",
    "r": "ⲅгᴦⲅꭇꭈꮁｒ𝐫𝑟𝒓𝓇𝓻𝔯𝕣𝖗𝗋𝗿𝘳𝙧𝚛",
    "s": "ѕѕꜱꮪｓ𐑈𑣁𝐬𝑠𝒔𝓈𝓼𝔰𝕤𝖘𝗌𝘀𝘴𝙨𝚜",
    "t": "𝚝𝐭𝑡𝒕𝓉𝓽𝔱𝕥𝖙𝗍𝘁𝘵𝙩𝚝",
    "u": "սʋυսᴜꞟꭎꭒｕ𐓶𑣘𝐮𝑢𝒖𝓊𝓾𝔲𝕦𝖚𝗎𝘂𝘶𝙪",
    "v": "ѵνטᴠ",
    "w": "ԝɯѡԝաᴡ𝓌𝔀𝔴𝕨𝖜𝗐𝘄𝘸𝙬𝚠",
    "x": "××х⨯𝐱𝑥𝒙𝓍𝔁𝗑𝘅𝘹𝙭𝚡",
    "y": "ууүỿｙ𑣜𝐲𝑦𝒚𝓎𝔂𝕪𝗒𝘆𝘺𝙮𝚢𝛄𝛾𝜸𝝲𝞬",
    "z": "ᴢᴢꮓｚ𝐳𝑧𝒛𝓏𝔃𝗓𝘇𝘻𝙯",
}
homos_lo = {
    "A":2,
    'B':4,
    'C':2,
    'D':0,
    'E':2,
    'F':1,
    'G':3,
    'H':4,
    'I':6,
    'J':2,
    'K':2,
    'L':1,
    'M':3,
    'N':3,
    'O':0,
    'P':2,
    'Q':2,
    'R':2,
    'S':2,
    'T':3,
    'U':1,
    'V':3,
    'W':1,
    'X':2,
    'Y':4,
    'Z':1,
    "a":6,
    "b": 3,
    "c": 5,
    "d": 2,
    "e": 8,
    "f": 0,
    "g": 5,
    "h": 6,
    "i": 3,
    "j": 4,
    "k": 1,
    "l": 3,
    "m": 0,
    "n": 3,
    "o": 3,
    "p": 4,
    "q": 5,
    "r": 0,
    "s": 3,
    "t": 0,
    "u": 7,
    "v": 0,
    "w": -1,
    "x": 4,
    "y": 4,
    "z": 2,
}
homos_lo1 = {
    "A":2,
    'B':1,
    'C':1,
    'D':8,
    'E':0,
    'F':0,
    'G':1,
    'H':2,
    'I':1,
    'J':0,
    'K':0,
    'L':7,
    'M':2,
    'N':1,
    'O':11,
    'P':1,
    'Q':2,
    'R':10,
    'S':0,
    'T':1,
    'U':0,
    'V':0,
    'W':0,
    'X':0,
    'Y':0,
    'Z':0,
    "a":3,
    "b": 0,
    "c": 2,
    "d": 0,
    "e": 0,
    "f": 5,
    "g": 0,
    "h": 0,
    "i": 0,
    "j": 2,
    "k": 1,
    "l": 2,
    "m": 0,
    "n": 0,
    "o": 0,
    "p": 0,
    "q": 0,
    "r": 1,
    "s": 0,
    "t": 0,
    "u": 2,
    "v": 1,
    "w": 4,
    "x": 2,
    "y": 0,
    "z": 3,
}
homos_lo2 = {
    "A":0,
    'B':0,
    'C':0,
    'D':0,
    'E':0,
    'F':0,
    'G':0,
    'H':0,
    'I':0,
    'J':0,
    'K':0,
    'L':0,
    'M':0,
    'N':0,
    'O':0,
    'P':0,
    'Q':0,
    'R':0,
    'S':0,
    'T':0,
    'U':0,
    'V':0,
    'W':0,
    'X':0,
    'Y':0,
    'Z':0,
    "a":0,
    "b": 0,
    "c": 0,
    "d": 0,
    "e": 0,
    "f": 0,
    "g": 0,
    "h": 0,
    "i": 5,
    "j": 0,
    "k": 0,
    "l": 3,
    "m": 1,
    "n": 0,
    "o": 0,
    "p": 0,
    "q": 0,
    "r": 0,
    "s": 4,
    "t": 0,
    "u": 0,
    "v": 0,
    "w": 0,
    "x": 0,
    "y": 0,
    "z": 0,
}
import homoglyphs as hg
hgc = hg.Homoglyphs(categories=('CYRILLIC', ))
def find_homo(input_char):
    # input_char=input_char.lower()
    if input_char in homos:
        # return homos[input_char][0]
        return homos[input_char][homos_lo1[input_char]]
        # return homos[input_char][max(0, homos_lo[input_char]-1)]
    else:
        # random_char = random.choice(unprintable_char)
        return input_char

def homo_back(input_char, style='del'):
    for key in homos:
        if input_char==homos[key][0]:
            if style=='map':
                return key
            elif style=='del':
                return ''
    return input_char

def text_homo_back(text, style='del'):
    new_text=''
    for tmp_char in text:
        new_text=new_text+homo_back(tmp_char, style='del')
    return new_text

def to_string(inputs, step_char=' '):
    output_str=''
    for input in inputs:
        if isinstance(input, list) and len(input)>20:
            continue
        if isinstance(input, str):
            output_str+=input
        else:
            output_str+=str(input)
        output_str+=step_char
    return output_str[0:-1]

def load_json(file_path):
    with open(file_path, 'r') as file:
        dict_list=json.load(file)
    return dict_list

def load_jsonl(file_path):
    dict_list=[]
    with open(file_path, 'r') as file:
        for line in file:
            dict_list.append(json.loads(line))
    return dict_list

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    # json_data = json.dumps(data)
    # with open(file_path, 'w', encoding='utf-8') as file:
    #     file.write(json_data)


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for d in data:
            try:
                json.dump(d, file, ensure_ascii=False, indent=4)
            except:
                print('WARING')
                print(d)

def get_advtext_filename(
        attack_name='',
        dataset_name='',
        victim_name='',
        num_examples=0,
        file_type='.json'
    ):

    return '_'.join([
        attack_name, dataset_name.replace('/','_'), victim_name.replace('/','_'), str(num_examples)
    ])+file_type

def truncation(text, tokenizer=None, max_token_num=100, min_num=80):
    if tokenizer is not None:
        text_ids=tokenizer.encode(text)[1:-1]
        if len(text_ids)<min_num:
            return '', 0
        new_text=tokenizer.decode(text_ids[:max_token_num])
        token_num=len(text_ids[:max_token_num])
    else:
        tokens = text.split()
        if len(tokens)<min_num:
            return '', 0
        sub_tokens=tokens[:max_token_num]
        token_num=len(sub_tokens)
        new_text = " ".join(sub_tokens)
    return new_text, token_num

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info', when='D',backCount=3, screen=True):
        self.logger = logging.getLogger(filename)
        self.logger.handlers.clear()
        # fmt_file='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        fmt_file='%(asctime)s - %(levelname)s: %(message)s'
        t_format_str = logging.Formatter(fmt_file)#设置日志格式

        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(t_format_str)#设置文件里写入的格式
        self.logger.addHandler(th)
        
        if screen:
            s_fmt_file='%(message)s'
            s_format_str = logging.Formatter(s_fmt_file)#设置日志格式
            sh = logging.StreamHandler()#往屏幕上输出
            sh.setFormatter(s_format_str) #设置屏幕上显示的格式
            sh.addFilter(self._screen_filter)  # Add custom filter to screen handler
            self.logger.addHandler(sh) #把对象加到logger里

    @staticmethod
    def _screen_filter(record):
        return len(record.getMessage()) <= 200

if __name__=='__main__':
    a=[]
    for idx in range (10):
        a.append({
            '1':idx,
            '2':str(idx)
        })
        if idx==4:
            a[-1]['3']=True
    save_json(a, 'saved_data/tmp.jspn')
    a=load_json('saved_data/tmp.jspn')
    print()