import os

import torch

from src.cli import get_args
from src.models.CNN import CNN
from src.models.MLP import MLP
from src.dataloader import get_dataLoader, CTWLabelledDataset
from src.trainer import Trainer

from datetime import datetime

# CUDA_VISIBLE_DEVICES=1 python main.py -mn mlp -lr 0.001 -ep 100
# python main.py -mn mlp -lr 0.001 -ep 100 -cu 1

if __name__ == "__main__":
    args = get_args()
    lr = args['learning_rate']
    batch_size = args['batch_size']
    epochs = args['epoch']
    weight_decay = args['weight_decay']
    print_iter = args['print_iter']
    model_name = args['model_name']
    patience = args['patience']
    mode = args['mode']
    model_path = args['model_path']

    # random seed
    torch.manual_seed(777)

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get model
    if model_name == 'cnn':
        model = CNN()
    elif model_name == 'mlp':
        model = MLP()
    else:
        print('Currently we are not supporting', model_name)
        raise NotImplementedError

    model = model.to(device)

    now = datetime.now()
    print("TRAIN START TIME: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

    dataLoader_saved = True
    # get dataloaders
    dataloaders = get_dataLoader(dataLoader_saved, bs=batch_size, use_concat=args['unlabelled'])

    # optimizer & loss function
    # TODO: later allow command line option to choose optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.L1Loss()

    # save logging file
    log_file = model_name + '_lr_' + str(lr) + '_bs_' + str(batch_size) + '_ep_' + str(epochs)

    # trainer
    trainer = Trainer(model=model,
                      epochs=epochs,
                      dataloaders=dataloaders,
                      optimizer=optimizer,
                      criterion=loss_fn,
                      device=device,
                      print_iter=print_iter,
                      log_file=log_file,
                      patience=patience
                    )

    if mode == 'train':
        model = trainer.train()

        now = datetime.now()
        print("TEST START TIME: {}".format(now.strftime("%d/%m/%Y %H:%M:%S")))

        trainer.test(model)

    elif mode == 'eval':
        if model_path is None:
            print('You have to specifi the model path!')
            exit(1)
        if not os.path.exists(model_path):
            print(f'Model path does not exist: {model_path}')
            exit(1)

        trainer.load_model_and_test(model_path=model_path)

    elif mode == 'self_train':
        if model_path is None:
            print('You have to specifi the model path!')
            exit(1)
        if not os.path.exists(model_path):
            print(f'Model path does not exist: {model_path}')
            exit(1)

        trainer.self_train(model_path=model_path)

    else:
        print(f'Mode {mode} does not exists; please select from [train/eval]')
        raise NotImplementedError