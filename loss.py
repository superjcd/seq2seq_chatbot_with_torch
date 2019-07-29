import torch
from vocab import PREV, NEXT

BOS = '<bos>'
PAD = '<pad>'

def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输入是BOS
    dec_input = torch.tensor([NEXT.vocab.stoi[BOS]] * batch_size)
    # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0
    l = torch.tensor([0.0])
    for y in Y.permute(1,0): # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # 将PAD对应位置的掩码设成0, 原文这里是 y != out_vocab.stoi[EOS], 感觉有误
        mask = mask * (y != NEXT.vocab.stoi[PAD]).float()
    return l / num_not_pad_tokens