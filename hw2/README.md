National Taiwan University of Science and Technology
MBA 
StudentID : M11008019
Name: Wu Ying-Huan(吳英緩)

Applied Deep Learning Homework 2

1. python files
   a. Multiple Choice 
      (a) The preprocessing.py is the file for converting the test_file to SWAG pattern.
      (b) The mc_test.py is the file for reproducing the prediction of Multiple Choice.

   b. Question Answering 
      The qa_test.py, trainer_qa.py and utils_qa.py are the files for reproducing the prediction on kaggle competition below.
      URL: https://www.kaggle.com/competitions/ntu-adl-hw2-spring-2021

   c. The learning curve of QA model
      The qa_plot.py is the file for plotting the eval learning curve of QA model.
     
2. download.sh
   This file will download the best training models.

3. Reproduce the Competition outcomes(csv) 
   a. bash download.sh
   b. bash run.sh  

4. Reproduce the learning curve pictures(png)
   (a)download models
      wget https://www.dropbox.com/s/qi2maz4tqh9xbrr/qa_bert_model.zip?dl=1 -O qa_best_model.zip
      wget https://www.dropbox.com/s/1qdyr77tb3akg3e/qa_best_model.zip?dl=1 -O qa_bert_model.zip
      unzip qa_best_model.zip
      unzip qa_bert_model.zip
      rm  -f qa_best_model.zip
      rm  -f qa_bert_model.zip

   (b)produce the curve pictures
      python qa_plot.py --trainer_state1_path <trainer_state1_path> --trainer_state2_path <trainer_state2_path>

      <trainer_state1_path>: path to qa_best_model(RoBERTa-wwm-ext)
      <trainer_state2_path>: path to qa_bert_model(BERT)
    
