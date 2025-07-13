
zwsp = chr(0x200B)
zwj = chr(0x200D)
print(f"Hello{zwj}World")  # 输出时看不到，但字符串中存在这个字符
print(len("Hello" + zwj + "World"))

# unprintable_char=''.join([chr(i) for i in range(1000) if chr(i).isprintable()==False])[0:10]
# for i in range(1000):
#     if chr(i).isprintable()==False:
#         print(i, 'abc'+chr(i)+'edf')