'''
  定义包括模型、 损失函数以及训练函数
'''

import torch
from torch import nn
import torch.nn.functional as F
from vocab import PREV, NEXT, train_iter
from configs import BasicConfigs

bc = BasicConfigs()
BOS = '<bos>'
PAD = '<pad>'


__all__ = ["Encoder", "Decoder", "train"]


####  模型 : 包括编码器、 解码器、 注意力机制

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # 输入形状是(批量大小, 时间步数)。将输出互换样本维和时间步维
        embedding = self.embedding(inputs.long())
        return self.rnn(embedding, state)

    def begin_state(self):
        return None  # 隐藏态初始化为None时PyTorch会自动初始化为0



def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size, attention_size), # pytorch的linear只会对最后一维起作用
                          nn.Tanh(),
                          nn.Linear(attention_size, 1))
    return model


def attention_forward(model, enc_states, dec_state):
    """
    enc_states: (时间步数, 批量大小, 隐藏单元个数)
    dec_state: (批量大小, 隐藏单元个数)
    """
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)  # cat需要保证另外维度是相同的
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    print(e.shape)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    print(alpha.shape) #10*4*1 *10*4*8
    return (alpha * enc_states).sum(dim=0)  # 返回背景变量, sum可以缩维度


class Decoder(nn.Module):
    '''
     解码器
    '''
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2*num_hiddens, attention_size)
        # GRU的输入包含attention输出的c和实际输入, 所以尺寸是 2*embed_size
        self.rnn = nn.GRU(2*embed_size, num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs) #
        state: output
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[-1])
        print(c.shape)
        # 将嵌入后的输入和背景向量在特征维连结
        print(self.embedding(cur_input).shape)
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

def batch_loss(encoder, decoder, X, Y, loss):
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

def train(encoder, decoder, lr, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = train_iter
    for epoch in range(num_epochs):
        l_sum = 0.0
        for batch in data_iter:
            X = batch.prev.to(bc.device)
            Y = batch.next.squeeze(0).to(bc.device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


