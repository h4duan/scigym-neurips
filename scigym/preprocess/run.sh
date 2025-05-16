#!/bin/bash
#SBATCH -p ml
#SBATCH -q ml
#SBATCH -A ml
#SBATCH -N 1
#SBATCH -w overture,quartet[1-2],quartet[4-5],concerto[1-2]
#SBATCH -c 8
#SBATCH --mem=48G

python preprocess/find_simul_time.py
