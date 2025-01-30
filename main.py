# Description: Main file for running the architecture search algorithm

# Import Packages
import argparse

from nas.config import *
from nas.manager import *
from models.obfuscated_model import *



# Parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=42, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=120)
parser.add_argument('--init_lr', type=float, default=0.01) 
parser.add_argument('--lr_schedule_type', type=str, default='cosine')

parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'stl10'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--valid_size', type=int, default=256)

parser.add_argument('--opt_type', type=str, default='adam', choices=['adam','sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  
parser.add_argument('--no_nesterov', action='store_true')  
parser.add_argument('--weight_decay', type=float, default=0) 
parser.add_argument('--label_smoothing', type=float, default=0.0)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--distort_color', type=str, default='None', choices=['normal', 'strong', 'None'])

""" net config """

""" arch search algo and warmup """
parser.add_argument('--warmup_epochs', type=int, default=25)
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  
parser.add_argument('--arch_adam_eps', type=float, default=1e-8) 
parser.add_argument('--arch_weight_decay', type=float, default=0)

""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)

torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    # Stores the object's attributes and their values as a dictionary.
    if args.dataset == 'cifar10':
        run_config = Cifar10RunConfig(
            **args.__dict__ 
        )
    elif args.dataset == 'cifar100':    
        run_config = Cifar100RunConfig(
            **args.__dict__
        )
    elif args.dataset == 'stl10':
        run_config = STL10RunConfig(
            **args.__dict__
        )
    else:
        raise NotImplementedError
    
    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0
    
    # 1. Build the Network
    # Whole Layers
    # args.input_channels = [32, 96, 192, 128, 128]
    # args.output_channels = [32, 96, 192, 128, 128]
    # args.kernel_size = [11, 5, 3, 3, 3]
    # args.stride = [1, 1, 1, 1, 1]
    # args.padding = [5, 2, 1, 1, 1]
    
    # Sensitive layers: 0, 1, 2
    args.input_channels = [32, 96, 192]
    args.output_channels = [32, 96, 192]
    args.stride = [1, 1, 1]
    
    # args.conv_candidates = [
    #     'Identity',
    #     '1x1_Conv',
    #     '11x11_Conv',
    # ]

    args.conv_candidates = [
        'Identity',
        '1x1_Conv',
        '3x3_Conv',
        '5x5_Conv',
        '7x7_Conv',
    ]
    
    super_net = ObfuscationNet(
        conv_candidates=args.conv_candidates,
        in_channels=args.input_channels,
        out_channels=args.output_channels,
        stride=args.stride,
        num_classes=run_config.data_provider.n_classes
    )
    
    # 2. Build arch search
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    arch_search_config = RLArchSearchConfig(**args.__dict__)
    
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))
    
    # 3. Arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)
    
    # 4. Resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            print("home: ", home)
            warmup_path = os.path.join(
                home, '/Test/checkpoint/arch_search/warmup.pth.tar'
            )
            if os.path.exists(warmup_path):
                print('load warmup weights')
                arch_search_run_manager.load_model(model_fname=warmup_path)
            else:
                print('fail to load models')
    
    # 5. Joint training
    start = time.time()
    arch_search_run_manager.train(fix_net_weights=args.debug)
    end = time.time()
    print('Time cost: %.2fs' % (end - start))
    