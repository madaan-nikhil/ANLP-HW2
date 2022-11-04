## **ANLP HW2 - Scientific Entity Recogntion**

# Results
| Model          | Data     | O test Acc    | O test F1     | Test F1       |
|------------------------------------|---------------------|------------------------------------|---------------------|---------------------|
| SciBERT frozen LM (Baseline) (A)      | Ours               | 53.20               | 0.12               | xx               |   
| SciBERT fine-tuned (B)      | Ours          | 94.23               | 0.72              | 0.4899  |              
| SciBERT fine-tuned (C)     | Ours + [IBM](https://aclanthology.org/2021.eacl-main.59.pdf)   | 86.124               | 0.545              | 0.411 |                
| SciBERT alternate (D)      | Ours          | 88.30              | 0.512              | xx   |             
| SciBERT FocalLoss (E)      | Ours          | 80.04              | 0.42              | 0.21 |


    *Check for istallation by Opening up a Python prompt by running the following:
		python train.py --model_dir '<model_save_dir>' --data_dir '<path to conll file>' --loss_weights '<class_wise_weights_for_loss_func>' --alternate_training 0




# Team Members
* Nikhil Madaan
* Shubhranshu Singh
* Rohan Panda
