#!/bin/bash

model=$1
compression=$2
echo $model
echo $compression
yhrun -N $SLURM_NNODES -n $SLURM_NNODES hostname | sort | sed 's/$/& slots=4/g' > hostlist
mpirun --mca btl openib,self,vader --mca btl_openib_if_include mlx5_2 \
 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_IB_DISABLE=1 \
 -bind-to none -map-by slot \
 --hostfile hostlist -np $SLURM_NTASKS \
 python trainer.py \
 -a $model >> ./log_txt/${model}_${compression}_${SLURM_NTASKS}.txt
