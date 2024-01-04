# LIVECell Segmentation - Improving Generalizability Using State-of-the-Art Data Augmentation Techniques Big Aug and Faster AutoAugment 

## Training
In order to train the model make sure to complete the following:

1. In `constants.py` choose the parameters you want to train U-Net with
2. CD to the parent folder
3. Run `python3 train.py`

## Results
The model was trained on 4 cell types: MCF7, BT474, BV2, 172. The model was tested for generalizability on 2 cell types: SHSY5Y, SkBr3. 
![Test Images](https://github.com/jessicapetrochuk/LIVECell_Segmentation/blob/main/images/train_test_cells.png)

After training, example segmentation results can be found below:
![Segmentation Images](https://github.com/jessicapetrochuk/LIVECell_Segmentation/blob/main/images/results.png)