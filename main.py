import logging
import torch
import torch.nn as nn
from logging import StreamHandler, FileHandler
from argparse import ArgumentParser
from modules import *
from vocab import prepare_vocab
from configs import BasicConfigs

bc = BasicConfigs()

# 定义日志输出功能
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 添加file_handeler
file_handler = FileHandler("log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
# 添加stream_handler
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'evaluate'],
                                      help='Choose mode to train or just use the model')
    parser.add_argument('--num_epochs', default=bc.num_epochs, help='Epochs for train')
    parser.add_argument('--embed_size', default=bc.embed_size, help='Emmbeding size')
    parser.add_argument('--num_hiddens', default=bc.num_hiddens, help='Hidden size')
    parser.add_argument('--num_layers', default=bc.num_layers, help='Layers of hidden state')
    parser.add_argument('--drop_prob', default=bc.drop_prob, help='Drop rates')
    parser.add_argument('--attention_size', default=bc.attention_size, help='Attention_size')
    parser.add_argument('--lr', default=bc.lr, help='Learning rate')
    parser.add_argument('--save_every', type=int, required=True, help='Save model after every [ ] epochs')
    parser.add_argument('--load_model_dir', type=str, help='Load model from the directory befor trainning')
    args = parser.parse_args()
    logger.info(f'We are going  to use following arguments:\n {args.__dict__}')
    PREV, NEXT, train_iter, val_iter= prepare_vocab()
    embedding = nn.Embedding(len(PREV.vocab), args.embed_size)
    encoder = Encoder(embedding, args.embed_size, args.num_hiddens, args.num_layers,
                      args.drop_prob)
    decoder = Decoder(embedding, len(PREV.vocab), args.embed_size, args.num_hiddens, args.num_layers,
                      args.attention_size, args.drop_prob)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    if args.load_model_dir:
        checkpoint = torch.load(args.load_model_dir)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        # 如果选择加载模型， 就直接将参数传入进去
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        enc_optimizer.load_state_dict(encoder_optimizer_sd)
        dec_optimizer.load_state_dict(decoder_optimizer_sd)
        encoder.embedding.load_state_dict(embedding_sd)
        decoder.embedding.load_state_dict(embedding_sd)
    if args.mode == 'train':
        logger.info('Start to  train the model')
        train(encoder, decoder, train_iter, enc_optimizer, dec_optimizer, args.num_epochs, args.save_every, NEXT)
    else:
        logger.info('Preparing model to use')
        # response
