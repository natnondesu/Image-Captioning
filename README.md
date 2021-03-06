# Image Captioning

Pattern Recognition 2/2021 Course Project 

## Dataset

Download Flicker8K dataset : [Here](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

Download train, val and test split (JSON file) : [Here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) by Andrej Karpathy

## Implementation

You can find python implementation in [source](https://github.com/natnondesu/Image-Captioning/tree/master/source) folder.

`utils.py` contains

- `prepare_data()` function which create ***train, validation, test*** split from Karpathy's JSON file
- `Create_image_caption_pair()` function which processing caption word to tokenized version and pair with its image.

  For example,

    Caption: `Professor Dang is playing double guitar` 

    SOS, EOS, PADDING, UNK: `<start> Professor Dang is playing double guitar <end> <pad> <pad> . . .`

    Tokenized: `3366 59 3368 1 30 120 152 3367 0 0 0 . . .`
    
`Dataset.py` contains

- `ImgCaption_Dataset` class which inherit from pytorch dataset. This class will Load image and transform it to Tensor and return as (x, y) pair.


## Baseline model

Using pretrained efficientNetb0 as image Encoder and GRU as Text Decoder.

- Prediction on validation image

![image](https://user-images.githubusercontent.com/62899961/161792324-d36e3278-1b3c-415a-82ea-14db8352c011.png)

## Decoder with Attention model (Show, Attend and Tell (2015) [Paper](https://arxiv.org/abs/1502.03044))

Bahdanau Attention (Additive) at decoder stage.

- Prediction on validation image

<p align="center">
  <img src="https://user-images.githubusercontent.com/62899961/163429536-af5c3bac-f065-476e-801f-9f18ce34c8a8.png" width="700" height="270">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/62899961/163430470-92d3ec75-7873-4325-9401-c0bee78ec002.png" width="700" height="270">
</p>

