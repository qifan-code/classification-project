# classification-project
This project is aiming for practice and adjust different parameters for classification methods
I use both R and Python code here. 

For train $ test split method, I am trying to keep same ratio for each class in training, testing and original dataset. Overall, Class0 has ratio about 44%, Class1 has ratio about 55% on original, training and testing dataset. 

To improve accuracy, I tried to normalize all numeric features in pre-processing step. I use outlier detection methods (1.5IQR rule in Python, boxplot in R). After classification, I create a table for comparasion. 

To get overall accuracy, I first use training & testing dataset for parameter adjustment. Then use cross validation with entire dataset to obtain the average accuracy. 

For classification, I basically use "Decision Tree", "Random Forest", "Bagging", "Boosting", "KSVM", "Naive Bayes" and "KNN". 

Here is my conclusion table: 
R code
	               data with Outlier	    Data without outlier
Model	        Accuracy	  Sensitivity	  Accuracy	  Sensitivity
Decision Tree	0.8344243	  0.875	        0.8174800	  0.8384
Random Forest	0.8791448	  0.8925	      0.8732531	  0.932
Bagging	      0.8431558	  0.868	        0.8375526	  0.941
Boosting	    0.8652174	  0.882	        0.8644068	  0.879
KSVM	        0.866	      0.786	        0.8533	    0.842
Naïve Bayes	  0.8638318	  0.8298	      0.8518027	  0.798
KNN	          0.8869565	  0.874	        0.8926554	  0.909

Python code
	            data with Outlier	        data without outlier
Model	        Accuracy	  Sensitivity	  Accuracy	  Sensitivity
Decision Tree	0.818986	  0.8819	      0.866778	  0.9091
Random Forest	0.870781	  0.9213	      0.880974	  0.909
Bagging	      0.877398	  0.921	        0.876587	  0.788
Boosting	    0.859810	  0.882	        0.876587	  0.879
KSVM	        0.871256	  0.898	        0.873574	  0.939
Naïve Bayes	  0.851520	  0.83	        0.845039	  0.80
KNN	          0.859294	  0.874	        0.877300	  0.909


Based on Accuracy and sensitivity, Random Forest performs the best with accuracy 88%. 
I will keep working on this dataset to improve accuracy. 
The best accuracy I can see on Internet for this dataset is 90%. 



Reference: 
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
