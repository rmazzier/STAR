# Attention Refined Unrolling for Sparse Micro-Doppler Reconstruction
This repository contains the implementation of the experimental results described in: `R. Mazzieri, J. Pegoraro and M. Rossi, "Attention-Refined Unrolling for Sparse Sequential micro-Doppler Reconstruction," in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2024.3408041.` (available [here](https://ieeexplore.ieee.org/abstract/document/10543012)).

If you find our work and implementation useful for your research, please, cite our work using the following BibTeX:
```
@article{10543012,
  author={Mazzieri, Riccardo and Pegoraro, Jacopo and Rossi, Michele},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Attention-Refined Unrolling for Sparse Sequential micro-Doppler Reconstruction}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  keywords={Stars;Sensors;Channel estimation;Compressed sensing;Spectrogram;Real-time systems;Human activity recognition;Joint Communication and Sensing;Micro-Doppler signatures;Sparse Reconstruction;Algorithm Unrolling;Attention;gHuman Activity Recognition},
  doi={10.1109/JSTSP.2024.3408041}}
```
## 1. Main Results (Section VI-C and VI-D)

STAR was tested on the DISC dataset, publicly available [here](https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems), in particular on the `uniform_7subj.zip` subset.


Our results can be reproduced by training STAR from scratch. To do so:
1. Configure the `ablation_runs.csv` file, by specifying possibly multiple configurations of the hyperparameters.
2. Train all the models specified in the .csv file by executing the following script:
```
python3 train.py
```
By default the `ablation_runs.csv` file contains all the ablation runs reported in the final paper. These can be modified to implement further experiments.


## 2. Results in novel environment (Sec VI-E)

[This folder](https://drive.google.com/drive/u/2/folders/19ev0y4MtivC2RE8QVyxSZmtNYn-rh7yB) contains the additional set of data (to be added to the DISC dataset in the short term)
To reproduce the results of the new environment, first train the final model as in step 1, and then execute:
```
python3 STAR_test_room.py
```


