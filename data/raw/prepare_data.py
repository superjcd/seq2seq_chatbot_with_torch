import pandas as pd
import re
from sklearn.model_selection import train_test_split

# set random seed
SEED = 727
TRAIN_RATIO = 0.95  # 尽量使用更多的数据进行训练

# 脏数据列表
noise_word = '\n|{.*?}|'


def sentence_cleaner(sentence):
    '''
    清除sentence中的脏数据
    :param sentence:
    :return:
    '''
    # 不要脏数据
    sentence = re.sub(noise_word, '', sentence)
    # 只要字符
    sentence = re.findall(r'\w+', sentence)
    return ''.join(sentence)


# 用于生成dataframe的数据集
out = {'prev':[], 'next':[]}


with open('青云语料.csv', encoding='utf-8') as file:
    for line in file:
        if len(line)<50:  #  把长度超过50的都去掉
            prev, next = line.split('|')[:2]
            out['prev'].append(sentence_cleaner(prev))
            out['next'].append((sentence_cleaner(next)))



data = pd.DataFrame(out)
train, test = train_test_split(data, train_size=TRAIN_RATIO, random_state=727)

# 保存数据
train.to_csv('../train.csv', index=False)
test.to_csv('../test.csv', index=False)


# if __name__ == '__main__':
#     target = '我是你爸爸， 你怎么回事***{*77%}'
#     print(sentence_cleaner(target))