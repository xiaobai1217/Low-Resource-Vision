# [Low-Resource Vision Challenges for Foundation Models](https://arxiv.org/pdf/2401.04716.pdf) (CVPR 2024)

[Yunhua Zhang](https://xiaobai1217.github.io/),  &nbsp  [Hazel Doughty](https://hazeldoughty.github.io/),  &nbsp  [Cees G.M. Snoek](https://www.ceessnoek.info/)

<img width="400" alt="Screenshot 2024-04-19 at 10 41 58" src="https://github.com/xiaobai1217/Low-Resource-Vision/assets/22721775/3676a97b-0052-40a5-8951-df442fcb6fe8">



[Website](https://xiaobai1217.github.io/Low-Resource-Vision/) [Dataset](https://uvaauas.figshare.com/articles/dataset/Low-Resource_Image_Transfer_Evaluation_Benchmark/25577145)

## Zero-Shot Transfer Evaluation

**Remember to change the data paths in them**. 

* Circuit Diagram Classification:
``
python zero-shot/circuit.py
``

* Historic Map Retrieval:
``
python zero-shot/historic_maps.py
``

* Mechanical Drawing Retrieval:
``
python zero-shot/mechanical_drawings.py
``

## Linear Probe Evaluation

* Circuit Diagram Classification:
``
python linear_probe/circuit.py
``

## Training Our Baselines

### Before Training
**Generating Data for Data Scarcity**

* Using this repository: [stable diffusion](https://github.com/CompVis/stable-diffusion)
* Augmenting circuit diagrams as an example: put the python script ``img2img_circuit.py`` into its directory, modify the data path in it, and run it by ``python img2img_circuit.py --strength 0.3 --outdir /path-to-output/ --n_samples 5`` to augment the data, here set the strength as 0.6 for the label-breaking augmentations.
* The augmentations for historic maps and mechanical drawings are the same

### Training

**First change the data paths in dataloaders and training scripts**. 

* Circuit Diagram Classification:
``
python scripts/train_circuit.py
``

* Historic Map Retrieval:
``
python scripts/train_map.py
``

* Mechanical Drawing Retrieval:
``
python scripst/train_mechanical.py
``

**The optimal hyperparameters listed in parser.add_argument() could be different on different machines.**
