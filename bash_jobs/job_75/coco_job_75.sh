#!/bin/bash
#$ -S /bin/bash
#$ -N Coco_75
#$ -o /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_75/sge.out
#$ -e /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_75/sge.err
#$ -q fastjobs
/home.stud/kolarj55/miniconda3/envs/yolov5/bin/python -u -Walways /home.stud/kolarj55/detector_improve_iopainting/coco_cropping_and_detecting_object.py --job 75