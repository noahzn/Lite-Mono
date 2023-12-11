I have no time to clean up the code, but I have kept all the necessary lines in ‘litemono.py’ to run the pretraining.

To be short, the last layer of the encoder of Lite-Mono should have `1000` channels for the classification task. Please add these lines to your current model file. The `main.py` file will create the model from this file.


To train on a single machine with 2 GPUs, using the following command.

    python -m torch.distributed.launch --nproc_per_node=2 main.py --data_path data/imagenet/


Plese check the code if you want to change parameters such as epochs, learning rates, etc.