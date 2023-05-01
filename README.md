# code for replication of results for paper "Semantic Similarity Loss for Neural Source Code Summarization"

## Preparation
- Please create a directory called outdir and three directories inside outdir called histories, models, and predictions.
- Please download the model file from [link]() and put the files in config folder to your local directory histories and put the files in models folder to your local directory models if you want to finetune models with simile or bleu.
- Note that you need to put files in config folder to the same folder as the outdir argument in train.py

## Step 0 Dataset
- We use three dataset for our experiments.
  - Funcom-java: Le Clair et al. [Arxiv](https://arxiv.org/abs/1904.02660)
  - funcom-java-long: Bansal et al. [Data](https://github.com/aakashba/humanattn). Please download q90 data and extract it to /nfs/projects/funcom/data/javastmt/q90 or change the --data argument in train.py.
  - python

## Step 1 Train your model
- To train the use-seq model with the data and gpu options, run the following command. Note that transformer-base means the transformer model.

  ```
  python3 train.py --model-type={transformer-base | ast-attendgru | codegnngru | setransformer}-use-seq  --gpu=0 --batch-size=50 --data={your data path}
  ```

  For example, transformer-base model on ./mydata

  ```
  python3 train.py --model-type=transformer-base-use-seq  --gpu=0 --batch-size=50 --data=./mydata
  ```
  
- To finetune baseline model with simile or bleu for one epoch, run the following command.
  
   ```
   python3 train.py --model-type={transformer-base | ast-attendgru | codegnngru | setransformer}-{simile | bleu-base}  --gpu=0 --batch-size=50 --epochs=1 --data={your data path} --load-model --model-file={your model path}
   ```
   
   For example, fintuning transformer-base model with simile on ./mydata with the model file called transformer.h5 on ./mymodel
    ```
    python3 train.py --model-type=transformer-base-simile  --gpu=0 --batch-size=50 --epochs=1 --data=./mydata --load-model --model-file=./mymodel/transformer.h5
    ```   
## Step 2 Predictions
- Once your training procedure is done, you can see the scrren with the accuracy on validation set. Pick the one before the biggest drop on validation accuracy. After you decide the model, run the following command to generate the prediction files.

  ```
  python3 predict.py {path to your model} --gpu=0 --data={your data path}
  ```
  
  For example, if your model path is outdir/models/transformer-base.h5 and your data path is ./mydata, run the following command.
  
  ```
  python3 predict.py outdir/models/transformer-base.h5 --gpu=0 --data=./mydata
  ```
## Step 3 Metrics
- We provide scripts for calculating the metrics that we report on the paper. 
  
 


  
 
