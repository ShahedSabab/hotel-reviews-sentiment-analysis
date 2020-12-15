# Hotel Reviews Sentiment Analysis
This project aims to classify ratings of the hotel reviews. There are 5 ratings (i.e., class) in the dataset along with the reviews. The dataset is quite balanced among the 5 classes. The objective is to develop two different classification models i.e., a baseline and a state of the art(SOTA) model to compare the performance of classifying ratings. To uncover interesting insights from the data, different exploratory analyses have been performed. Also, the dataset includes unstructured data (i.e., reviews) which need further conversion (e.g.,vectorization) before this can be used in any machine learning model. Different approaches have been followed for developing the models.

Baseline Model (TF-IDF + Logistic Regression): The baseline model consists of a simple vectorization (i.e., feature engineering) approach using TF-IDF. The resultant vectorized texts are then used for classification using Logistic Regression. Overall, 75.8 % accuracy is achieved from this model.   

SOTA Model (RoBERTa): The SOTA model leverages transfer learning and a custom RoBERTa model to classify the reviews. Overall, 79% accuracy is achieved from this model.

• The base model uses Tf-Idf and logistic regression.
• Pytorch is used to develop the SOTA model.
• The SOTA model uses Hugging face 'roberta-base' pretrained model.
• Dropout has been performed to reduce overfitting.
• The best model(SOTA) achieves 79% accuracy.

# Performance
<img src="class_distribution.PNG" width="60%">
<img src="length_distribution.PNG" width="60%">

<img src="rating1_words.PNG" width="50%">
<img src="rating3_words.PNG" width="50%">
<img src="rating5_words.PNG" width="50%">
<img src="performance_lr.PNG" width="50%">

# How to run:
Baseline model: Please check the hotel_reviews_sentiment_baseline.ipynb file for the detailed analysis. The trained baseline model can be loaded using the following command:

> pickle.load('model_baseline.pkl')

SOTA model: Please check the hotel_reviews_sentiment_sota.ipynb file for the detailed analysis. The trained sota model can be loaded using the following command:

> torch.load('model_sota.pb')
