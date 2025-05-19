#!/bin/bash
conda activate htgnn

python process.py
python preprocess.py
python run_icews.py 
