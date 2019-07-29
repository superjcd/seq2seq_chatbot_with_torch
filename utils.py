import jieba
import pkuseg

# 分词工具
seg = pkuseg.pkuseg(user_dict='dictionary/special.lex')

# 定义一个tokenizer
def chi_tokenizer(sentence):
    return seg.cut(sentence)





if __name__ == '__main__':
    print(seg.cut('<bos>你妈贵姓<eos>'))