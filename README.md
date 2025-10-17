# 🥷 Scamurai-Shield: Scam Detector

This model can detector whether a message is scam or not



### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Train Model

   ```
   $ python Model/theModel.py
   ```

3. Run app

   ```
   $ streamlit run Model/forntend.py
   ```   

### Features 

- Detects Scam messages
- Use TF-IDF and Logistic regression
- Build with Streamlit
- SMOTE and RandomOverSampler for balancing data

### Accuracy
- This model is in it's early stages
- Higher chance that the model will give wrong output 
- Still working on better ways to improve model

### Future Improvements
- Better UI
- Using transformer models(eg. BERT)
- Improvements in accurary

### Dataset
- [SMS Spam Collection, from Kaggle](https://www.kaggle.com/datasets/noorsaeed/scam-detection-dataset/data?source=post_page-----bcd84e36e689---------------------------------------)

