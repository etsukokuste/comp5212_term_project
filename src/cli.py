import argparse

def get_args():
    parser = argparse.ArgumentParser(description='COMP5212 Term Project')
    # Device
    parser.add_argument('-cu', '--cuda', help='CUDA', type=str, required=False, default='0')

    # Mode
    parser.add_argument('-mo', '--mode', help='Mode (train/eval/self_train)', type=str, required=False, default='train')
    parser.add_argument('-un', '--unlabelled', help='Use unlabelled dataset', action='store_true')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=False, default=32)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=False, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0)
    parser.add_argument('-ep', '--epoch',help='Epoch', type=int, required=False, default=10)
    parser.add_argument('-pa', '--patience', help='Patience to stop training', type=int, required=False, default=5)

    parser.add_argument('-pr', '--print-iter', help='Print every X iterations during training', type=int,
                        required=False, default=100)

    # model selection
    parser.add_argument('-mn', '--model-name', type=str, help='Model name e.g., cnn, mlp...', required=True)

    # model path (required in inference)
    parser.add_argument('-mp', '--model-path', help='Path to the saved model', type=str, required=False, default=None)

    args = vars(parser.parse_args())
    return args