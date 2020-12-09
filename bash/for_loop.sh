#!/bin/bash
# Basic range with steps for loop

#beta_array=(0.01 0.1 0.001)
#beta_array=(0.0001 0.01)
beta_array=(0.0001)

for beta in ${beta_array[@]};
do
#  echo $beta
#   python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --beta=$beta
   python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot3way/ --num_classes=3 --beta=$beta

   echo Finish one experiment

done

echo All done