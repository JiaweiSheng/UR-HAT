import argparse


def str2bool(str):
    return True if str.lower() == 'true' else False


def parse_args():
    parser = argparse.ArgumentParser(description='train a neural network for document-level event factuality prediction')

    # 存储
    parser.add_argument('--data_path', type=str, default='./data/chinese.xml', help='path to the data file')
    parser.add_argument('--saved_path', type=str, default='./data/', help='path to the saved_data file')

    parser.add_argument('--model_name_or_path', type=str, default='bert-base-chinese', help='pre-trained language model')
    parser.add_argument('--path', type=str, default="./result")

    # 训练
    parser.add_argument('--n_epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='size of the training batches')
    parser.add_argument('--labmda', type=float, default=0.2)
    parser.add_argument('--gpu',
                        dest="gpu",
                        action="store_const",
                        const=True,
                        default=True,
                        required=False,
                        help='optional flag to use GPU if available')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    # 模型结构
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--gcn_layers', type=int, default=2, help="the number of gcn layers")
    parser.add_argument('--gcn_hid_dim', type=int, default=768, help='the hidden size of gcn')
    parser.add_argument('--gcn_out_dim', type=int, default=768, help='the output size of gcn')
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--linear_dim', type=int, default=300)

    # 调试
    parser.add_argument('--variant', type=str, default='origin', help='')
    parser.add_argument('--exp_name', type=str, default='default', help='')
    parser.add_argument("--kl_loss", default=True, help="", type=str2bool)
    parser.add_argument("--aux_loss", default=True, help="", type=str2bool)
    parser.add_argument("--gamma", default=0.001, type=float, help="")

    parser.add_argument('--max_sen_num', type=int, default=35)
    parser.add_argument('--max_tri_num', type=int, default=20)
    parser.add_argument('--max_sen_len', type=int, default=80)
    parser.add_argument('--max_keyword_len', type=int, default=10)

    parser.add_argument('--language', type=str, default='en')

    parser.add_argument("--force_refresh", default=False, help="", type=str2bool)
    parser.add_argument("--exp_edge", type=str)

    parser.add_argument('--keyword_ratio', type=float, default=0.3)
    parser.add_argument("--norm_adj", default=False, help="", type=str2bool)
    parser.add_argument("--norm_type", default='row', help="", type=str)

    parser.add_argument("--training", default=True, help="", type=str2bool)
    parser.add_argument('--seed', type=int, default=2040)

    parser.add_argument("--aba", default='entire', type=str)
    parser.add_argument('--n_rel', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=2)

    parser.add_argument('--alpha', type=float, default=0.001)

    args = parser.parse_args()
    for arg in vars(args):
        print('{}={}'.format(arg.upper(), getattr(args, arg)))
    print('')
    return args


config = parse_args()
