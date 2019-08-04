'''
  定义包括模型、 损失函数以及训练函数
'''
import os
import  logging
import torch
from torch import nn
import torch.nn.functional as F
from configs import BasicConfigs
from utils import chi_tokenizer
from tqdm import tqdm


__all__ = ["Encoder", "Decoder", "train", "translate"]


logger = logging.getLogger()

bc = BasicConfigs()
BOS = '<bos>'
PAD = '<pad>'
EOS = '<eos>'


####  模型 : 包括编码器、 解码器、 注意力机制

class Encoder(nn.Module):
    def __init__(self, embedding, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = embedding
        #self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        embedding = self.embedding(inputs.long())
        return self.rnn(embedding, state)

    def begin_state(self):
        return None  # 隐藏态初始化为None时PyTorch会自动初始化为0



def attention_model(input_size, attention_size):
    '''
     这里有一个很有意思的现象在于， 原论文使用了线性组合来计算编码阶段输出和解码器的隐藏阶段之间的相似度。
     其实是可以达到dot production的效果：因为只有两个向量（最后一维度）符号相近的时候，
     在相的时候才能取得较大的绝对值（当权重参数w的符号相等时）
    :param input_size:
    :param attention_size:
    :return:
    '''
    model = nn.Sequential(nn.Linear(input_size, attention_size), # pytorch的linear只会对最后一维起作用
                          nn.Tanh(),
                          nn.Linear(attention_size, 1))
    return model


def attention_forward(model, enc_states, dec_state):
    """
    enc_states: (时间步数, 批量大小, 隐藏单元个数) ->outputs
    dec_state: (批量大小, 隐藏单元个数)
    """
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)  # cat需要保证另外维度是相同的
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=0)  # 返回背景变量, sum可以缩维度


class Decoder(nn.Module):
    '''
     解码器
    '''
    def __init__(self, embedding, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = embedding
        self.attention = attention_model(2*num_hiddens, attention_size)
        # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 2*embed_size
        self.rnn = nn.GRU((embed_size+num_hiddens), num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        注意这里的enc_states, 在实际计算的时候，使用的是编码器的outputs, 另外state是编码器的layers
        """
        # 使用注意力机制计算背景向量
        # print(cur_input.shape, state.shape, enc_states.shape)
        c = attention_forward(self.attention, enc_states, state[-1]) # 所以state[-1]只得是最后一层
        # 将嵌入后的输入和背景向量在特征维连结
        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1) # (批量大小, 2*embed_size)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state



####  损失函数 ####

def batch_loss(encoder, decoder, X, Y, loss, NEXT):
    '''

    :param encoder:
    :param decoder:
    :param X:
    :param Y:
    :param loss:  loss function
    :param NEXT:  下一个句子
    :return:
    '''
    # print(f'X的形状为：{X.shape}, Y的形状为：{Y.shape}')
    batch_size = X.shape[1]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是BOS
    dec_input = torch.tensor([NEXT.vocab.stoi[BOS]] * batch_size)
    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0
    l = torch.tensor([0.0])
    for y in Y:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # 将PAD对应位置的掩码设成0, 原文这里是 y != out_vocab.stoi[EOS], 感觉有误
        mask = mask * (y != NEXT.vocab.stoi[PAD]).float()
    return l / num_not_pad_tokens


#### 训练函数 ####

def train(encoder, decoder, train_iter, enc_optimizer, dec_optimizer, num_epochs, save_every, NEXT):
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = train_iter
    for epoch in tqdm(range(num_epochs)):
        logger.info('Epoch num is {}'.format(epoch+1))
        l_sum = 0.0
        for batch in data_iter:
            X = batch.prev.to(bc.device)
            Y = batch.next.squeeze(0).to(bc.device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss, NEXT)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 5 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
        # 保存模型
        if (epoch +1) % save_every == 0:
            logger.info('save the modle')
            directory = os.path.join('models', '{}-{}'.format(
                                                            encoder.num_layers,
                                                            encoder.num_hiddens))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'epochs':epoch,
                'en':encoder.state_dict(),
                'de':decoder.state_dict(),
                'en_opt':enc_optimizer.state_dict(),
                'de_opt':dec_optimizer.state_dict(),
                'embedding':encoder.embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))


####   贪婪搜索  ###
def response(encoder, decoder, PREV, NEXT,input_seq, max_seq_len):
    in_tokens = chi_tokenizer(input_seq)
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.tensor([[PREV.vocab.stoi[tk] for tk in in_tokens]]) # batch=1
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([NEXT.vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = NEXT.vocab.itos[int(pred.item())]
        if pred_token == EOS:  # 当任一时间步搜索出EOS时，输出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens





