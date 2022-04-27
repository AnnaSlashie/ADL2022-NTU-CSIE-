# Applied Deep Learning Homework 2

## 1. python files <br>
   * a. Multiple Choice <br>
      (a) The preprocessing.py is the file for converting the test_file to SWAG pattern. <br>
      (b) The mc_test.py is the file for reproducing the prediction of Multiple Choice. <br>

   * b. Question Answering <br>
      The qa_test.py, trainer_qa.py and utils_qa.py are the files for reproducing the prediction on kaggle competition below. <br>
      URL: https://www.kaggle.com/competitions/ntu-adl-hw2-spring-2021 <br>

   * c. The learning curve of QA model <br>
      The qa_plot.py is the file for plotting the eval learning curve of QA model. <br>

## 2. download.sh <br>
   This file will download the best training models. <br>
   
## 3. Reproduce the Competition outcomes(csv) <br>
   * a. bash download.sh <br>
   * b. bash run.sh  <br>

## 4. Reproduce the competition outcomes(csv.files) 
   * a. bash download.sh
   * b. bash intent_cls.sh  <dataset_path> <pred_csv_path>
   * c. bash slot_tag.sh  <dataset_path> <pred_csv_path> <br>
        <dataset_path>: path to dataset <br>
         <pred_csv_path>: path to predict csv <br>
   
## 4. Reproduce the learning curve pictures(png) <br>
   * a. download models <br> 
        wget https://www.dropbox.com/s/qi2maz4tqh9xbrr/qa_bert_model.zip?dl=1 -O qa_best_model.zip <br>
        wget https://www.dropbox.com/s/1qdyr77tb3akg3e/qa_best_model.zip?dl=1 -O qa_bert_model.zip <br>
        unzip qa_best_model.zip <br>
        unzip qa_bert_model.zip <br>
        rm  -f qa_best_model.zip <br>
        rm  -f qa_bert_model.zip <br>
      
   * b. produce the curve pictures <br>
      * python qa_plot.py --trainer_state1_path <trainer_state1_path> --trainer_state2_path <trainer_state2_path> <br>
       <br>
        <trainer_state1_path>: path to qa_best_model(RoBERTa-wwm-ext) <br>
        <trainer_state2_path>: path to qa_bert_model(BERT) <br>
