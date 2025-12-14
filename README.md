# CSCI-3302-Final-Project
Trained LLM (BERT) for sentiment analysis.


Dependencies:
- Python 3.10+
- PyTorch 
- Transformers (HuggingFace)
- Dataset (HuggingFace)
- Scikit-learn
- NumPy
  
  
Running environment:

1.) Download/clone repository and create and activate a virtual environment:  
  
Windows (PowerShell):  
python -m venv venv  
venv\Scripts\activate  
  
macOS / Linux:  
python3 -m venv venv  
source venv/bin/activate  

  
2.) Install dependencies:  
pip install --upgrade pip  
pip install torch transformers datasets scikit-learn evaluate numpy  

  
3.) Run training script:  
python train.py  

  
4.) Run testing script:  
python evaluate.py  

  
5.) Run cross validation:  
python cross_validate.py  
  

6.) You can live test the trained model on custom text input byrunning:  
python predict.py "INSERT TEXT HERE"  


7.) ROC graph generation to visualize performance:  
python plot_roc.py  
  

8.) Error analysis (false positives/negatives):  
python find_errors.py  
