![Screenshot](figure.png)

## Training command example

```
python -u train.py --dropout_rate=0.3 --epoch=1000 --ngpu=1 --batch_size=256 --num_workers=0
```
We added only about data of 1000 samples in data folder due to size of dataset so the performance is much lower than the paper
Each sample is saved in pikcle file and it consists of two rdkit objects of a ligand and protein.
The inputs of neural network are processed in the dataset class on the fly. 

