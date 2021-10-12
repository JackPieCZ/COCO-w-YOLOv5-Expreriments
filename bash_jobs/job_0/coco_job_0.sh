#!/bin/bash
#$ -S /bin/bash
#$ -N Coco_0
#$ -o /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_0/sge.out
#$ -e /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_0/sge.err
#$ -q offline
/home.stud/kolarj55/miniconda3/envs/yolov5/bin/python -u -Walways /home.stud/kolarj55/detector_improve_iopainting/coco_cropping_and_detecting_object.py --job 0