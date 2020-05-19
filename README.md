# Comp5212 Term Project: Deep Learning based Signal Processing for Wireless Communication

## Getting Started
Please read this carefully and follow the steps one-by-one

### Prerequisites
```
$ pip install -r requirements.txt
```

### Prepare Labelled-Data and Split and save them into three .pkl files (train, valid, test)
For this, only (src/data.py) is used and this code file will not be used again in any further experiment.
Please do not change seed number in data.py becuase it's for consistency. The seed number should stay as 77.

1. Open terminal and make folder 'data'
```
$ mkdir data
```
2. Copy your unzipped labelled CTW2020_labelled_data folder inside data (./data/CTW2020_labelled_data)
3. To run data.py code, locate your shell to directory of src.
```
$ cd src
```
4. Run data.py file to read and split data into Train, Valid, Test sets
```
$ python data.py
```
- Then, three .pkl files will be generated in data folder.
- Each pickle file has data in dictionary format of {'H_Re': H_Re, 'H_Im': H_Im, 'SNR': SNR, 'Pos': Pos}
- The data is too huge, so for development stage, we just decided used one file of data ("../data/CTW2020_labelled_data/file_1.hdf5") for default setting. So, once you run the code, you will see output on shell. 
>[TOTAL: 512] TRAIN: 409, VAL: 51, TEST: 52
- On the training stage, we need all_data. All data is supposed to be splitted [Train(80%): 3,983, Valid(10%): 498, Test(10%): 498]

5. Done! :D 
- You don't need to do anything with these .pkl files (will be handled in dataloader automatically)

### Understanding dataloader
Once data .pkl files are ready, you can use pytorch dataloader. You just need to call function `$get_dataLoader(batch_size)`. The default batch size is 32. Output will be a dictionary, so you need to use as following:

```
dataloaders = get_dataLoader(bs=batch_size)
train_data = dataloaders['train']
valid_data = dataloaders['valid']
test_data = dataloaders['test']
```

### main.py
#### Train a model
You can train a model by 
```python main.py -mn cnn```.
You can also specify other hyperparameters e.g., batch size, learning rate.  To see the other command line options, please 
```python main.py -h```
All those command line options are implemented in src/cli.py.

#### Understanding main.py
While main.py manages the whole training procedure, the actual training is done by Trainer, which is in src/trainer. We get command line options to specify necessary arguments for Trainer, and pass them to Trainer. You do not have to write training codes for each model since main.py and src/trainer.py do not depend on them. Trainer takes care of the training, evaluation, logging, etc.


## Miscellaneous Information
`links.txt` includes miscellaneous information such as a link to our presentation video on YouTube.


