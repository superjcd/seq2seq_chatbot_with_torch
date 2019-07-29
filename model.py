import torch
from torch import nn
import torch.nn.functional as F


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
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[-1]) # 把encoder的最后一层layer的隐藏层给取出来
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




if __name__ == '__main__':
    pass
