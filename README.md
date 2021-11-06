# DLoc code

This repository has the MATLAB data pre-processing and the PyTorch implementation of DLoc from [Deep Learning based Wireless Localization for Indoor Navigation](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894). 

The datasets required to run these codes can be downloaded from the [WILD](https://wcsng.ucsd.edu/wild/) website.

## MATLAB Codes
These codes contain the pre-processing codes to convert the raw Channel-State-Information that is downloaded from the [WILD](https://wcsng.ucsd.edu/wild/) website to the 2D XY heatmap images shown in the paper.

For further details please refer to the [README](./MATLAB/README.md) file in the MATLAB folder.

## PyTorch Codes

To install requirements:

```setup
pip install -r requirements.txt
```


## Training and Evlautaion

To train the model(s) in the paper and evaluate them, run this command:

```train
python train_and_test.py
```

The file automatically imports the parameters from [params.py](params.py).

The parameters and their descriptions can be found in the comments of the example implementaion of the [params.py](params.py) file.

To recreate the results from the [paper](https://dl.acm.org/doi/pdf/10.1145/3372224.3380894) refer to the [README](./params_storage/README.md) of the *params_storage* folder.


<!-- ## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->
