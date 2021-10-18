# #!/bin/bash
# #$ -S /bin/bash
# #$ -N Coco_0
# #$ -o /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_0/sge.out
# #$ -e /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_0/sge.err
# #$ -q offline
# /home.stud/kolarj55/miniconda3/envs/yolov5/bin/python -u -Walways /home.stud/kolarj55/detector_improve_iopainting/coco_cropping_and_detecting_object.py --class 0

import os

target_dir = ('bash_jobs')
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

jobs_count = int(input('Jobs count: '))
for i in range(jobs_count):
    job_dir = os.path.join(target_dir, f'job_{i}')
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)
    job_file = os.path.join(job_dir, f'coco_job_{i}.sh')
    with open(job_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#$ -S /bin/bash\n')
        f.write(f'#$ -N Coco_{i}\n')
        f.write(f'#$ -o /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_{i}/sge.out\n')
        f.write(f'#$ -e /mnt/home.stud/kolarj55/detector_improve_iopainting/bash_jobs/job_{i}/sge.err\n')
        f.write('#$ -q offline\n')
        f.write(f'/home.stud/kolarj55/miniconda3/envs/yolov5/bin/python -u -Walways /home.stud/kolarj55/detector_improve_iopainting/coco_cropping_and_detecting_object.py --class {i}')
