from torchtext.data import Field, TabularDataset, BucketIterator
from utils import chi_tokenizer
from configs import BasicConfigs

bc = BasicConfigs()

# 定义Field
PREV = Field(tokenize=chi_tokenizer, init_token='<bos>', eos_token='<eos>') # 在这例可以添加很多有用的参数， 比如pa_token，unknowntoken,stopwords
NEXT = Field(tokenize=chi_tokenizer, init_token='<bos>', eos_token='<eos>')

# 定义字段与FIELD之间读配对
fields = [('prev', PREV), ('next',NEXT)]

# 注意skip_header
train, val = TabularDataset.splits(path='data',train='train.csv',
                                   validation='test.csv',
                                   format='csv',
                                   fields=fields,
                                   skip_header=True)



# 构建vocabulary时同时使用到了train， 和val的数据
PREV.build_vocab(train, val)
NEXT.build_vocab(train, val)
#  需要注意的是， PREV和NEXT的字典是不一样的

# 定义数据生成器
train_iter = BucketIterator(train, batch_size=bc.batch_size, \
sort_key=lambda x: len(x.prev), sort_within_batch=True, shuffle=True)

val_iter = BucketIterator(val, batch_size=bc.batch_size, \
sort_key=lambda x: len(x.prev), sort_within_batch=True, shuffle=True)



if __name__ =='__main__':
    print(PREV.vocab.itos[10], NEXT.vocab.itos[10], sep='-')
    for i, data in enumerate(val_iter):
        if i <3:
            print(data.prev.shape)
            print(data.prev)
        else:
            break
        print('*' * 10)












