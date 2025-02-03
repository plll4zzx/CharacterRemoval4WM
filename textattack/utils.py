import json
import logging
from logging import handlers
import random

unprintable_char=''.join([chr(i) for i in range(1000) if chr(i).isprintable()==False])[0:10]
def find_homo(input_char):
    homos = {
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
        "a": "ɑ",
        "b": "Ь",
        "c": "ϲ",
        "d": "ԁ",
        "e": "е",
        "f": "𝚏",
        "g": "ɡ",
        "h": "հ",
        "i": "і",
        "j": "ϳ",
        "k": "𝒌",
        "l": "ⅼ",
        "m": "ｍ",
        "n": "ո",
        "o": "о",
        "p": "р",
        "q": "ԛ",
        "r": "ⲅ",
        "s": "ѕ",
        "t": "𝚝",
        "u": "ս",
        "v": "ѵ",
        "w": "ԝ",
        "x": "×",
        "y": "у",
        "z": "ᴢ",
    }
    input_char=input_char.lower()
    if input_char in homos:
        return homos[input_char]
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