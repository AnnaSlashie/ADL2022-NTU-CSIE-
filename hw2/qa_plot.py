import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
import os.path
from pathlib import Path 

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--trainer_state1_path', help='Path to the trainer-state1.',default='./qa_best_model/trainer_state.json')
    parser.add_argument('--trainer_state2_path', help='Path to the trainer-state2.',default='./qa_bert_model/trainer_state.json')
    args = parser.parse_args()
    return args

#load trainer_state
trainer_state1 = pd.read_json(args.trainer_state1)['log_history']
trainer_state1 = pd.read_json(trainer_state1.to_json(orient='records'))

trainer_state2 = pd.read_json(args.trainer_state2)['log_history']
trainer_state2 = pd.read_json(trainer_state2.to_json(orient='records'))

#pick out eval_loss and eval_exact_match
trainer_state1=trainer_state1[['step','eval_loss','eval_exact_match']]
trainer_state2=trainer_state2[['step','eval_loss','eval_exact_match']]

drop_index1=[i for i in range(len(trainer_state1)) if i%2==0]
drop_index2=[i for i in range(len(trainer_state2)) if i%2==0]

trainer_state1=trainer_state1.drop(drop_index1)
trainer_state2=trainer_state2.drop(drop_index2)

trainer_state1.set_index('step', inplace=True) 
trainer_state2.set_index('step', inplace=True) 

#Start to plot the eval learning curve of QA model
plt.figure(figsize=(10,5))
plt.plot(trainer_state1[['eval_loss']],'o-',color = 'blue', label="roberta")
plt.plot(trainer_state2[['eval_loss']],'o-',color = 'lightblue', label="bert-base-chinese")
plt.title("ADLHW2_QA_eval_loss",fontweight='bold')
plt.xlabel('step')
plt.ylabel('eval_loss')
plt.legend(loc = "best")
plt.savefig('ADLHW2_QA_eval_loss.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(trainer_state1[['eval_exact_match']],'o-',color = 'blue', label="roberta")
plt.plot(trainer_state2[['eval_exact_match']],'o-',color = 'lightblue', label="bert-base-chinese")
plt.title("ADLHW2_QA_eval_exact_match",fontweight='bold')
plt.xlabel('step')
plt.ylabel('exact_match')
plt.legend(loc = "best")
plt.savefig('ADLHW2_QA_eval_exact_match.png')
plt.show()
