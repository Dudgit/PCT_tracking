#! /usr/bin/bash
source /home/bdudas/anaconda3/lib/python3.9/site-packages/conda/shell/etc/profile.d/conda.sh
conda activate pct
nohup jupyter notebook --no-browser --port 9999 >/dev/null 2>&1 &
echo "Jupyter notebook is running on port 9999"
sleep 5
jupyter server list