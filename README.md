![GitHub contributors](https://img.shields.io/github/contributors/MendelinaL/Capstone)
![Project](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)
![Code](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

# Capstone: Hate Speech Detection Through Sentiment Analysis
***
This is a capstone project for the M.S. in Applied Data Science Program with the University of San Diego.

# Installation
To use this project, please follow these instructions:

git init

git clone https://github.com/MendelinaL/Capstone.git

# Contributors
[Katie Hu](https://github.com/katie-hu) <br>
[Mendelina Lopez](https://github.com/MendelinaL) <br>
[John Chen](https://github.com/jjchen-SEA) <br>

# Method
- Exploratory Data Analysis
- Preprocessing
- Modeling
- Evaluation

# Project Objective
The dataset the team used focuses specifically on Twitter posts taken from Kaggle. To kick things off the team will gain initial insight through exploratory data analysis and data cleaning. This project is set to convey to what extent offensive language is a problem and the possible growth at which it could go. It is obvious that hate speech is an absolute issue that needs to be mitigated, the team is interested in addressing the matter and further flagging users who might show an increase in negative posts. The objective of this sentiment analysis project is to detect hate speech and offensive language in tweets using past posts from Twitter. If the findings prove to be substantial, a hypothetical consulting group should be interested in creating a safer space for users online.

# Applicable Visuals
![alt text](https://github.com/MendelinaL/Capstone/blob/main/Image/Exploratory%20Data%20Analysis/hate_word_cloud.png)
> Word cloud plots provide a visual representation of what is included in the text data. The figure above exemplifies the types of negative words that are considered hate speech. The terms displayed are the top common words used within hate speech tweets with the larger words being the highest concentration. This is a clear representation of hate speech as the words are commonly flagged as racist, homophobic, sexist, and offensive.

![alt text](https://github.com/MendelinaL/Capstone/blob/main/Image/Exploratory%20Data%20Analysis/tweet_length_density_plot.png)
> At this time, tweet posts hold a max of 10,000 characters in length. The above plot gives the variation of tweet length within the dataset. As reflected, the graph shows a curve slightly skewed to the right with a median of around 10 characters.

![alt text](https://github.com/MendelinaL/Capstone/blob/main/Image/Exploratory%20Data%20Analysis/sentiment_distribution.png)
> Sentiment Analysis is a way for data scientists to determine the attitude behind a given piece of text. The graph shows the sentiment distribution of tweets. It is apparent that the majority of posts contain a sentiment of 0.00. This is with the remaining ratios having a respective frequency of around 2,000.

# Model Evaluation
| Model | Accuracy | AUC | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bernoulli Naive-Bayes | 0.60 | 0.88 | 0.75 | 0.60 | 1.00 |
| Multinomial Naive Bayes | 0.61 | 0.90 | 0.75 | 0.60 | 1.00 |
| XGBoost Classifier | 0.80 | 0.93 | 0.88 | 0.92 | 0.84 |
| Logistic Regression | 0.87 | 0.94 | 0.92 | 0.91 | 0.93 |
| LSTM | 0.90 | 0.96 | 0.94 | 0.94 | 0.93 |

![image](https://github.com/MendelinaL/Capstone/assets/102394762/dc5060db-8abe-4662-9ef0-871bec59f954)

# Conclusion
The project’s fundamental premise was the belief that a hate speech detection system for social media platforms could be realized through NLP techniques and capabilities of machine learning models. In conclusion, our working hypothesis is  confirmed using the potential of the application that was created where a tweet or sentence can be tested for positive, negative, and neutral sentiment at an accuracy rate of about 83% the team has paved a way for a more proactive approach in countering hate speech and offensive language. As the team strives to create a safer and more inclusive online community, the tools and insights developed can be leveraged for positive change on social network applications. 

# Acknowledgments
The team would like to thank Dr. Ebrahim Tarshizi for his helpful direction throughout the process of this project. As well as another special thank you to Dr. Roozbeh Sadeghian for his support in the feature engineering phase of the process.
