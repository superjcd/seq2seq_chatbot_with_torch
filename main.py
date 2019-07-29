import logging
from logging import StreamHandler, FileHandler
from argparse import ArgumentParser
from modules import *
from vocab import PREV, NEXT
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
    parser.add_argument('--num_epochs', default=bc.num_epochs, help='Epochs for train')
    parser.add_argument('--embed_size', default=bc.embed_size, help='Emmbeding size')
    parser.add_argument('--num_hiddens', default=bc.num_hiddens, help='Hidden size')
    parser.add_argument('--num_layers', default=bc.num_layers, help='Layers of hidden state')
    parser.add_argument('--drop_prob', default=bc.drop_prob, help='Drop rates')
    parser.add_argument('--attention_size', default=bc.attention_size, help='Attention_size')
    parser.add_argument('--lr', default=bc.lr, help='Learning rate')
    args = parser.parse_args()
    print(args)
    encoder = Encoder(len(PREV.vocab), args.embed_size, args.num_hiddens, args.num_layers,
                      args.drop_prob)
    decoder = Decoder(len(NEXT.vocab), args.embed_size, args.num_hiddens, args.num_layers,
                      args.attention_size, args.drop_prob)

    ## start to train the model
    train(encoder, decoder, args.lr, args.num_epochs)
