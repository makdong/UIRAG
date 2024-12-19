#!/bin/sh

#SBATCH -J preprocess_wiki
#SBATCH -o /home/dhkim0317/C4RAG/script/wiki.log #
#SBATCH -e /home/dhkim0317/C4RAG/script/error.log
#SBATCH --time 3-00:00:00

#SBATCH -p cpu-max24 

#SBATCH -q nogpu
#SBATCH -N 1
#SBATCH -n 24

#SBATCH --comment wikipreprocess

numGPU=4

# echo "numGPU $numGPU"

echo "Activate conda"
source /home/dhkim0317/anaconda3/etc/profile.d/conda.sh
conda activate C4RAG

echo "Call start"
chmod 755 /home/dhkim0317/C4RAG/script/new.sh
/home/dhkim0317/C4RAG/script/new.sh

echo "Call end"
