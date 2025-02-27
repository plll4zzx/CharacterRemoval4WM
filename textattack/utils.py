import json
import logging
from logging import handlers
import random

unprintable_char=''.join([chr(i) for i in range(1000) if chr(i).isprintable()==False])[0:10]
homos = {
    "A":"ΑАᎪᗅᴀꓮꭺＡ𐊠𖽀𝐀𝐴𝑨𝒜𝓐𝔸𝖠𝗔𝘈𝘼𝙰𝚨𝛢𝜜𝝖𝞐",
    'B':'ʙΒВвᏴᏼᗷᛒꓐＢ𐊂𐊡𐌁𝐁𝐵𝑩𝓑𝔅𝔹𝕭𝖡𝗕𝘉𝘽𝙱𝚩𝛣𝜝𝝗𝞑',
    'C':'ϹСᏟᑕℂⅭ⊂Ⲥ⸦ꓚＣ𐊢𐌂𐐕𝐂𝐶𝑪𝒞𝓒𝖢𝗖𝘊𝘾𝙲🝌',
    'D':'ᎠᗞᗪᴅⅅⅮꓓꭰＤ𝐃𝐷𝑫𝒟𝓓𝔇𝔻𝕯𝖣𝗗𝘋𝘿𝙳',
    'E':'ΕЕᎬᴇ⋿ⴹꓰꭼＥ𐊆𑢦𑢮𝐄𝐸𝑬𝔼𝖤𝗘𝘌𝙀𝙴𝚬𝛦𝜠𝝚𝞔',
    'F':'ϜᖴꓝＦ𐊇𐊥𐔥𑢢𝈓𝐅𝐹𝑭𝓕𝔽𝖥𝗙𝘍𝙁𝙵𝟊',
    'G':'ɢԌԍᏀᏳᏻꓖꮐＧ𝐆𝐺𝑮𝔾𝖦𝗚𝘎𝙂𝙶',
    'H':'ʜΗНнᎻᕼℋℌℍⲎꓧꮋＨ𐋏𝐇𝐻𝑯𝓗𝖧𝗛𝘏𝙃𝙷𝚮𝛨𝜢𝝜𝞖',
    'I':'ɪΙІӀӏⅠⲒＩ𝐈𝐥𝐼𝑰𝕀𝗹𝙡𝙸𝚕𝚰𝛪𝜤',
    'J':'ͿЈᎫᒍᴊꓙꞲꭻＪ𝐉𝐽𝑱𝒥𝓙𝕁𝖩𝗝𝘑𝙅𝙹',
    'K':'ΚКᏦᛕKⲔꓗＫ𐔘𝐊𝐾𝑲𝒦𝓚𝕂𝖪𝗞𝘒𝙆𝙺𝚱𝛫𝜥𝝟𝞙',
    'L':'ʟᏞᒪℒⅬⳐⳑꓡꮮＬ𐐛𐑃𐔦𑢣𑢲𖼖𝈪𝐋𝐿𝑳𝓛𝕃𝖫𝗟𝘓𝙇𝙻',
    'M':'ΜϺМᎷᗰᛖℳⅯⲘꓟＭ𐊰𐌑𝐌𝑀𝑴𝓜𝔐𝕄𝖬𝗠𝘔𝙈𝙼𝚳𝛭𝜧𝝡𝞛',
    'N':'ɴΝℕⲚꓠＮ𐔓𝐍𝑁𝑵𝒩𝓝𝔑𝕹𝖭𝗡𝘕𝙉𝙽𝚴𝛮𝜨𝝢𝞜',
    'O':'𐊒𐊫𐐄𐓂𑢵𑣠𝐎𝑂𝑶𝒪𝓞𝖮𝗢𝘖𝙊𝙾𝚶𝛰𝜪𝝤𝞞',
    'P':'ΡРᏢᑭᴘᴩℙⲢꓑꮲＰ𐊕𝐏𝑃𝑷𝒫𝓟𝔓𝕻𝖯𝗣𝘗𝙋𝙿𝚸𝛲𝜬𝝦𝞠',
    'Q':'ℚⵕＱ𝐐𝑄𝑸𝒬𝓠𝔔𝕼𝖰𝗤𝘘𝙌𝚀',
    'R':'ƦʀᎡᏒᖇᚱℛℜℝꓣꭱꮢＲ𐒴𖼵𝈖𝐑𝑅𝑹𝓡𝕽𝖱𝗥𝘙𝙍𝚁',
    'S':'ЅՏᏕᏚꓢＳ𐊖𐐠𖼺𝐒𝑆𝑺𝒮𝓢𝔖𝕊𝕾𝖲𝗦𝘚𝙎𝚂',
    'T':'ΤТтᎢᴛ⊤⟙ⲦꓔꭲＴ𐊗𐊱𐌕𑢼𖼊𝐓𝑇𝑻𝒯𝓣𝔗𝕋𝕿𝖳𝗧𝘛𝙏𝚃𝚻𝛕𝛵𝜏𝜯𝝉𝝩𝞃𝞣𝞽🝨',
    'U':'Սሀᑌ∪⋃ꓴＵ𐓎𑢸𖽂𝐔𝑈𝑼𝒰𝓤𝔘𝕌𝖀𝖴𝗨𝘜𝙐𝚄',
    'V':'Ѵ٧۷ᏙᐯⅤⴸꓦꛟＶ𐔝𑢠𖼈𝈍𝐕𝑉𝑽𝒱𝓥𝕍𝖵𝗩𝘝𝙑𝚅',
    'W':'ԜᎳᏔꓪＷ𑣦𑣯𝐖𝑊𝑾𝒲𝓦𝕎𝖶𝗪𝘞𝙒𝚆',
    'X':'ΧХ᙭ᚷⅩ╳ⲬⵝꓫꞳＸ𐊐𐊴𐌗𐌢𐔧𑣬𝐗𝑋𝑿𝒳𝓧𝕏𝖷𝗫𝘟𝙓𝚇𝚾𝛸𝜲𝝬𝞦',
    'Y':'ΥϒУҮᎩᎽⲨꓬＹ𐊲𑢤𖽃𝐘𝑌𝒀𝒴𝓨𝕐𝖸𝗬𝘠𝙔𝚈𝚼𝛶𝜰𝝪𝞤',
    'Z':'ΖᏃℤꓜＺ𐋵𑢩𑣥𝐙𝑍𝒁𝒵𝓩𝖹𝗭𝘡𝙕𝚉𝚭𝛧𝜡𝝛𝞕',
    "a": "ɑɑαа⍺ａ𝐚𝑎𝒂𝒶𝓪𝔞𝕒𝖆𝖺𝗮𝘢𝙖𝚊𝛂𝛼𝜶𝝰𝞪",
    "b": "ЬƄЬᏏᑲᖯｂ𝐛𝑏𝒃𝔟𝕓𝖇𝖻𝗯𝘣𝙗𝚋",
    "c": "ϲϲсᴄⅽⲥꮯｃ𐐽𝐜𝑐𝒄𝒸𝓬𝔠𝕔𝖈𝖼𝗰𝘤𝙘𝚌",
    "d": "ԁԁᏧᑯⅆⅾꓒｄ𝐝𝑑𝒅𝒹𝓭𝖽𝗱𝘥𝙙𝚍",
    "e": "ееҽ℮ℯⅇꬲｅ𝐞𝑒𝒆𝓮𝔢𝕖𝖊𝖾𝗲𝘦𝙚𝚎",
    "f": "𝚏ϝẝꞙꬵｆ𝐟𝑓𝒇𝔣𝕗𝖋𝖿𝗳𝘧𝙛𝚏𝟋",
    "g": "ɡɡցᶃｇ𝐠𝑔𝒈𝓰𝔤𝕘𝖌𝗀𝗴𝘨𝙜𝚐",
    "h": "հһհᏂℎｈ𝐡𝒉𝒽𝓱𝕙𝖍𝗁𝗵𝘩𝙝𝚑",
    "i": "іiıｉ𝐢𝑖𝒊𝒾𝓲𝔦𝖎𝗂𝗶𝘪𝙞𝚒",
    "j": "ϳϳјⅉｊ𝐣𝑗𝒋𝒿𝓳𝔧𝕛𝖏𝗃𝗷𝘫𝙟𝚓",
    "k": "𝒌ｋ𝐤𝑘𝒌𝓀𝓴𝔨𝕜𝖐𝗄𝗸𝘬𝙠𝚔",
    "l": "ⅼ|ƖᛁℓⅼⲒⵏꓲ𝚕𝝞𝞘",
    "m": "ｍⅿ",
    "n": "ոոռｎ𝐧𝑛𝒏𝓃𝓷𝔫𝕟𝖓𝗇𝗻𝘯𝙣𝚗",
    "o": "оοо𐐬𐓪𐔖𑓐𝐨𝑜𝒐𝕠𝖔𝗈𝗼𝘰𝙤𝚘𝛐𝜊𝝄𝝾𝞸",
    "p": "рρр⍴ⲣｐ𝐩𝑝𝒑𝓅𝓹𝔭𝕡𝖕𝗉𝗽𝘱𝙥𝚙𝛒𝝆𝞀𝞺",
    "q": "ԛԛգզｑ𝐪𝑞𝒒𝓆𝓺𝔮𝕢𝖖𝗊𝗾𝘲𝙦𝚚",
    "r": "ⲅгᴦⲅꭇꭈꮁｒ𝐫𝑟𝒓𝓇𝓻𝔯𝕣𝖗𝗋𝗿𝘳𝙧𝚛",
    "s": "ѕѕꜱꮪｓ𐑈𑣁𝐬𝑠𝒔𝓈𝓼𝔰𝕤𝖘𝗌𝘀𝘴𝙨𝚜",
    "t": "𝚝𝐭𝑡𝒕𝓉𝓽𝔱𝕥𝖙𝗍𝘁𝘵𝙩𝚝",
    "u": "սʋυսᴜꞟꭎꭒｕ𐓶𑣘𝐮𝑢𝒖𝓊𝓾𝔲𝕦𝖚𝗎𝘂𝘶𝙪",
    "v": "ѵνטᴠ",
    "w": "ԝɯѡԝաᴡ𝓌𝔀𝔴𝕨𝖜𝗐𝘄𝘸𝙬𝚠",
    "x": "××х⨯𝐱𝑥𝒙𝓍𝔁𝗑𝘅𝘹𝙭𝚡",
    "y": "ууүỿｙ𑣜𝐲𝑦𝒚𝓎𝔂𝕪𝗒𝘆𝘺𝙮𝚢𝛄𝛾𝜸𝝲𝞬",
    "z": "ᴢᴢꮓｚ𝐳𝑧𝒛𝓏𝔃𝗓𝘇𝘻𝙯",
    # "-": "˗",
    # "9": "৭",
    # "8": "Ȣ",
    # "7": "𝟕",
    # "6": "б",
    # "5": "Ƽ",
    # "4": "Ꮞ",
    # "3": "Ʒ",
    # "2": "ᒿ",
    # "1": "l",
    # "0": "O",
    # "'": "`",
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
def find_homo(input_char):
    # input_char=input_char.lower()
    if input_char in homos:
        # return homos[input_char][0]
        return homos[input_char][homos_lo[input_char]]
    else:
        # random_char = random.choice(unprintable_char)
        return input_char

def to_string(inputs):
    output_str=''
    for input in inputs:
        if isinstance(input, list) and len(input)>20:
            continue
        if isinstance(input, str):
            output_str+=input
        else:
            output_str+=str(input)
        output_str+=' '
    return output_str

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

        s_fmt_file='%(message)s'
        s_format_str = logging.Formatter(s_fmt_file)#设置日志格式

        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(s_format_str) #设置屏幕上显示的格式
        sh.addFilter(self._screen_filter)  # Add custom filter to screen handler
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
        if screen:
            self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

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