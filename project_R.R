cat("\014") # clears console
rm(list = ls()) # clears global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE) # clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE) # clears packages
options(scipen = 100) # disables scientific notation for entire R session

library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(e1071)
library(magrittr)
library(pROC)
library(randomForest)
library(mclust)
library(Metrics)
library(ipred)
library(xgboost)
library(ggplot2)



setwd("C:/Users/39930/Desktop/ALY6040/project")
df <- read.csv("heart.csv")
df

str(df)
#################################################################################
#data cleaning
#################################################################################
df <- na.omit(df)
df
df$Sex <- as.factor(df$Sex)
df$ChestPainType <- as.factor(df$ChestPainType)
df$RestingECG <- as.factor(df$RestingECG)
df$ST_Slope <- as.factor(df$ST_Slope)
df$ExerciseAngina <- as.factor(df$ExerciseAngina)
df$HeartDisease <- as.factor(df$HeartDisease)

#################################################################################
#EDA
#################################################################################
boxplot(df)

#################################################################################
#normalization
#################################################################################
df[, c(4, 5, 8)] <- scale(select(df, c(RestingBP, Cholesterol, MaxHR)))
df

#################################################################################
#data splitting
#################################################################################
set.seed(1000)
class0 <- filter(df, df$HeartDisease == 0)
class0

class1 <- subset(df, !(HeartDisease %in% class0$HeartDisease))
class1

df_description <- matrix(
  c(nrow(class0), nrow(class1), nrow(class0)/nrow(df), nrow(class1)/nrow(df)), 
  nrow = 2,
  ncol = 2,
)
rownames(df_description) <- c("class0", "class1")
colnames(df_description) <- c("count", "ratio")
df_description


sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train <- rbind(train0, train1)
train <- train[sample(1:nrow(train)), ] #shuffle
train

test <- rbind(test0, test1)
test <- test[sample(1:nrow(test)), ] #shuffle
test

train_description <- matrix(
  c(nrow(train0), nrow(train1), nrow(train0)/nrow(train), nrow(train1)/nrow(train)), 
  nrow = 2,
  ncol = 2,
)
rownames(train_description) <- c("class0", "class1")
colnames(train_description) <- c("count", "ratio")
train_description

test_description <- matrix(
  c(nrow(test0), nrow(test1), nrow(test0)/nrow(test), nrow(test1)/nrow(test)), 
  nrow = 2,
  ncol = 2,
)
rownames(test_description) <- c("class0", "class1")
colnames(test_description) <- c("count", "ratio")
test_description
#################################################################################
#check if dataset is balanced
#################################################################################
library(GGally)
ggpairs(train, ggplot2::aes(color = HeartDisease, alpha = .4))


#################################################################################
#rebalance sample - oversampling or undersampling
#SMOTE or ADASYN 
#################################################################################


#################################################################################
#create different models
#################################################################################

#################################################################################
##decision tree
#################################################################################
decision_tree_model <- rpart(HeartDisease~., data = train, method = "class")
decision_tree_model
summary(decision_tree_model)
rpart.plot(decision_tree_model)
importance <- varImp(decision_tree_model)
importance %>% arrange(desc(Overall))

decision_tree_preds <- predict(decision_tree_model, newdata = test, type = "class")
decision_tree_preds

confusionMatrix(test$HeartDisease, decision_tree_preds)
#accuracy = 0.833 sensitivity = 0.8743

boxplot(df)
#remove outliers
df_temp <- df
outliers <- boxplot(df_temp$RestingBP, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$RestingBP %in% outliers),]

outliers <- boxplot(df_temp$Cholesterol, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Cholesterol %in% outliers),]

outliers <- boxplot(df_temp$Oldpeak, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Oldpeak %in% outliers),]
##########train&test
class0 <- filter(df_temp, df_temp$HeartDisease == 0)
class0

class1 <- subset(df_temp, !(HeartDisease %in% class0$HeartDisease))
class1

sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train_temp <- rbind(train0, train1)
train_temp <- train_temp[sample(1:nrow(train_temp)), ] #shuffle
train_temp

test_temp <- rbind(test0, test1)
test_temp <- test_temp[sample(1:nrow(test_temp)), ] #shuffle
test_temp

####decision tree
decision_tree_model <- rpart(HeartDisease~., data = train_temp, method = "class")
decision_tree_model
summary(decision_tree_model)
rpart.plot(decision_tree_model)
importance <- varImp(decision_tree_model)
importance %>% arrange(desc(Overall))

decision_tree_preds <- predict(decision_tree_model, newdata = test_temp, type = "class")
decision_tree_preds

confusionMatrix(test_temp$HeartDisease, decision_tree_preds)
#accuracy = 0.819 sensitivity = 0.8418

#cross validation
train_control <- trainControl(
  method = "cv",        # k-fold cross-validation
  number = 10,          # number of folds
  savePredictions = "final",
  verboseIter = FALSE
)
set.seed(123)  
model <- train(
  HeartDisease ~ .,          # formula (response ~ predictors)
  data = df,          # dataset
  method = "rpart",     # model type: recursive partitioning -> decision tree
  trControl = train_control  # control parameters
)

print(model)
confusionMatrix(model)
#accuracy = 0.8344

#no prune needed because it is not camplex tree

#cross validation
train_control <- trainControl(
  method = "cv",        # k-fold cross-validation
  number = 10,          # number of folds
  savePredictions = "final",
  verboseIter = FALSE
)
set.seed(123)  # for reproducibility
model <- train(
  HeartDisease ~ .,          # formula (response ~ predictors)
  data = df_temp,          # dataset
  method = "rpart",     # model type: recursive partitioning -> decision tree
  trControl = train_control  # control parameters
)

print(model)
confusionMatrix(model)
#accuracy = 0.8344

#################################################################################
##randomforest
#################################################################################
rf1 <- randomForest(HeartDisease~., data = train, ntree = 350, ntry = 9, importance = TRUE, na.action = randomForest::na.roughfix, replace = FALSE)
varImpPlot(rf1, col = 4)

random_forest_preds <- predict(rf1, newdata = test)
random_forest_preds

table_rf <- table(test$HeartDisease, random_forest_preds)
confusionMatrix(table_rf)
#accuracy = 0.8559

##randomforest-find best parameter
accuracy_data <- c()
ntree_data <- c()
ntry_data <- c()
sensitivity_data <- c()
for (i in c(seq(300, 800, 10))){
  for (j in c(seq(2, 10, 1))){
    rf1 <- randomForest(HeartDisease~., data = train, ntree = i, ntry = j, importance = TRUE, na.action = randomForest::na.roughfix, replace = FALSE)
    random_forest_preds <- predict(rf1, newdata = test)
    table_rf <- table(test$HeartDisease, random_forest_preds)
    confusion_rf <- confusionMatrix(table_rf)
    accuracy_data <- c(accuracy_data, confusion_rf$overall['Accuracy'])
    sensitivity_data <- c(sensitivity_data, confusion_rf$byClass['Sensitivity'])
    ntree_data <- c(ntree_data, i)
    ntry_data <- c(ntry_data, j)
  }
}

random_forest_accuracy_df <- data.frame(
  ntree = ntree_data,
  ntry = ntry_data,
  accuracy = accuracy_data, 
  recall = sensitivity_data
)
random_forest_accuracy_df <- random_forest_accuracy_df[order(-random_forest_accuracy_df$accuracy),]
random_forest_accuracy_df
#so the best choice here is ntree = 390, ntry = 8, accuracy = 0.8689956

#using cross validation
# Define train control
train_control <- trainControl(
  method = "cv",         # cross-validation
  number = 10            # number of folds
)
# Train the model
model <- train(
  df[, 1:11],
  df$HeartDisease,
  method = "rf",           # rf denotes Random Forest
  trControl = train_control,
  ntree = 360, 
  ntry = 4# number of trees
)
# Print model details
print(model)
# Summary of results
summary(model)

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 2.
rf1 <- randomForest(HeartDisease~., data = train_temp, ntree = 310, mtry = 2 , importance = TRUE, na.action = randomForest::na.roughfix, replace = FALSE)
varImpPlot(rf1, col = 4)

random_forest_preds <- predict(rf1, newdata = test)
random_forest_preds

table_rf <- table(test$HeartDisease, random_forest_preds)
confusionMatrix(table_rf)
#so the best choice here is ntree = 390, mtry = 2, accuracy = 0.8646

#conclusion:
#based on validation of parameters, ntree = 390, ntry = 8, accuracy = 0.8689956
#based on crossvalidation, ntree = 390, mtry = 2, accuracy = 0.8646
##randomforest-find best parameter
accuracy_data <- c()
ntree_data <- c()
ntry_data <- c()
sensitivity_data <- c()
for (i in c(seq(300, 800, 10))){
  for (j in c(seq(2, 10, 1))){
    rf1 <- randomForest(HeartDisease~., data = train_temp, ntree = i, ntry = j, importance = TRUE, na.action = randomForest::na.roughfix, replace = FALSE)
    random_forest_preds <- predict(rf1, newdata = test_temp)
    table_rf <- table(test_temp$HeartDisease, random_forest_preds)
    confusion_rf <- confusionMatrix(table_rf)
    accuracy_data <- c(accuracy_data, confusion_rf$overall['Accuracy'])
    sensitivity_data <- c(sensitivity_data, confusion_rf$byClass['Sensitivity'])
    ntree_data <- c(ntree_data, i)
    ntry_data <- c(ntry_data, j)
  }
}

random_forest_accuracy_df <- data.frame(
  ntree = ntree_data,
  ntry = ntry_data,
  accuracy = accuracy_data, 
  recall = sensitivity_data
)
random_forest_accuracy_df <- random_forest_accuracy_df[order(-random_forest_accuracy_df$accuracy),]
random_forest_accuracy_df

######################################
train_control <- trainControl(
  method = "cv",         # cross-validation
  number = 10            # number of folds
)
# Train the model
model <- train(
  df_temp[, 1:11],
  df_temp$HeartDisease,
  method = "rf",           # rf denotes Random Forest
  trControl = train_control,
  ntree = 300, 
  ntry = 6# number of trees
)
# Print model details
print(model)
# Summary of results
summary(model)
#################################################################################
##bagging
#################################################################################

bag1 <- bagging(formula = HeartDisease~., 
                data = train, 
                nbagg = 400, 
                coob = TRUE, 
                control = rpart.control(minsplit = 2, cp = 0, min_depth = 2))

bag1

bagging_pres <- predict(bag1, newdata = test)
bagging_pres

confusionMatrix(test$HeartDisease, bagging_pres)

##bagging-find best parameter
accuracy_data <- c()
sensitivity_data <- c()
n_bag_data <- c(seq(50, 400, 10))
for(i in n_bag_data){
  bag1 <- bagging(formula = HeartDisease~., 
                  data = train, 
                  nbagg = i, 
                  coob = TRUE, 
                  control = rpart.control(minsplit = 2, cp = 0, min_depth = 2))
  bagging_pres <- predict(bag1, newdata = test)
  confusion_bagging <- confusionMatrix(test$HeartDisease, bagging_pres)
  accuracy_data <- c(accuracy_data, confusion_bagging$overall['Accuracy'])
  sensitivity_data <- c(sensitivity_data, confusion_bagging$byClass['Sensitivity'])
}

bagging_accuracy_df <- data.frame(
  n_bag = n_bag_data,
  accuracy = accuracy_data, 
  recall = sensitivity_data
)
bagging_accuracy_df <- bagging_accuracy_df[order(-bagging_accuracy_df$accuracy),]
bagging_accuracy_df
#n_bags = 130, accracy = 0.8515284


#cross validation
bag2 <- train(
  HeartDisease ~ .,
  data = df,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 70,  
  control = rpart.control(minsplit = 2, cp = 0)
)
print(bag2) #try nbagg = best n_bags

##################################
#var importance
initial_model <- bagging(formula = HeartDisease~., 
                         data = train, 
                         nbagg = 140, 
                         coob = TRUE, 
                         control = rpart.control(minsplit = 2, cp = 0, min_depth = 2))

initial_pred <- predict(initial_model, test)
confusion_bagging <- confusionMatrix(test$HeartDisease, initial_pred)
initial_accuracy <- confusion_bagging$overall['Accuracy']

feature_performance <- data.frame(Feature=names(train[1:11]), Accuracy=numeric(length=11))
for (i in 1:11) {
  #print(i)
  feature_subset <- train[, -i, drop=FALSE]
  feature_subset$HeartDisease <- train$HeartDisease
  
  temp_model <- bagging(formula = HeartDisease~., 
                        data = feature_subset, 
                        nbagg = 140, 
                        coob = TRUE, 
                        control = rpart.control(minsplit = 2, cp = 0, min_depth = 2))
  temp_pred <- predict(temp_model, test[,-i, drop = FALSE])
  confusion_bagging <- confusionMatrix(table(test$HeartDisease, temp_pred))
  temp_accuracy <- confusion_bagging$overall['Accuracy']
  
  feature_performance$Accuracy[i] <- temp_accuracy
}

feature_performance$AccuracyDifference = initial_accuracy - feature_performance$Accuracy
print(feature_performance[order(feature_performance$AccuracyDifference),])









############################
##bagging-find best parameter
accuracy_data <- c()
sensitivity_data <- c()
n_bag_data <- c(seq(50, 400, 10))
for(i in n_bag_data){
  bag1 <- bagging(formula = HeartDisease~., 
                  data = train_temp, 
                  nbagg = i, 
                  coob = TRUE, 
                  control = rpart.control(minsplit = 2, cp = 0, min_depth = 2))
  bagging_pres <- predict(bag1, newdata = test_temp)
  confusion_bagging <- confusionMatrix(test_temp$HeartDisease, bagging_pres)
  accuracy_data <- c(accuracy_data, confusion_bagging$overall['Accuracy'])
  sensitivity_data <- c(sensitivity_data, confusion_bagging$byClass['Sensitivity'])
}

bagging_accuracy_df <- data.frame(
  n_bag = n_bag_data,
  accuracy = accuracy_data, 
  recall = sensitivity_data
)
bagging_accuracy_df <- bagging_accuracy_df[order(-bagging_accuracy_df$accuracy),]
bagging_accuracy_df
#n_bags = 130, accracy = 0.8515284


#cross validation
bag2 <- train(
  HeartDisease ~ .,
  data = df_temp,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 70,  
  control = rpart.control(minsplit = 2, cp = 0)
)
print(bag2) #try nbagg = best n_bags

#################################################################################
##boosting
#################################################################################

setwd("C:/Users/39930/Desktop/ALY6040/project")
df <- read.csv("heart.csv")
df
df <- na.omit(df)
df
df$Sex <- as.numeric(as.factor(df$Sex))
df$ChestPainType <- as.numeric(as.factor(df$ChestPainType))
df$RestingECG <- as.numeric(as.factor(df$RestingECG))
df$ST_Slope <- as.numeric(as.factor(df$ST_Slope))
df$ExerciseAngina <- as.numeric(as.factor(df$ExerciseAngina))
df$HeartDisease <- as.numeric(as.factor(df$HeartDisease))

df[, c(4, 5, 8)] <- scale(select(df, c(RestingBP, Cholesterol, MaxHR)))
df

set.seed(1000)
class0 <- filter(df, df$HeartDisease == 1)
class0

class1 <- subset(df, !(HeartDisease %in% class0$HeartDisease))
class1

sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train <- rbind(train0, train1)
train <- train[sample(1:nrow(train)), ] #shuffle
train

test <- rbind(test0, test1)
test <- test[sample(1:nrow(test)), ] #shuffle
test




train$HeartDisease <- ifelse(train$HeartDisease == 1, 0, 1)
test$HeartDisease <- ifelse(test$HeartDisease == 1, 0, 1)


boost1 <- xgboost(data = as.matrix(train[, -which(names(train) == "HeartDisease")]), 
                  label = train$HeartDisease,  
                  max.depth = 2,
                  eta = 1,
                  nthread = 5,
                  nrounds = 200,
                  objective = "binary:logistic",
                  verbose = 0  # Set verbose to 0 to reduce log output, or use 1 for more verbose
)
boost1

test_matrix <- as.matrix(test[, -which(names(test) == "HeartDisease")])
boosting_preds <- predict(boost1, test_matrix)
boosting_preds

actual_labels <- test$HeartDisease
predicted_labels <- ifelse(boosting_preds > 0.5, 1, 0)

confusionMatrix(as.factor(predicted_labels), as.factor(actual_labels))

#find best parameter
max_depth_lst <- c(seq(1, 11, 1))
nrounds_lst <- c(seq(100, 300, 100))
accuracy_data <- c()
depth_data <- c()
nrounds_data <- c()
info_loss_data <- c()
for (i in nrounds_lst){
  for (j in max_depth_lst){
    boost1 <- xgboost(data = as.matrix(train[, -which(names(train) == "HeartDisease")]), 
                      label = train$HeartDisease,  
                      max.depth = j,
                      eta = 1,
                      nthread = 5,
                      nrounds = i,
                      objective = "binary:logistic",
                      verbose = 0  # Set verbose to 0 to reduce log output, or use 1 for more verbose
    )
    test_matrix <- as.matrix(test[, -which(names(test) == "HeartDisease")])
    boosting_preds <- predict(boost1, test_matrix)
    actual_labels <- test$HeartDisease
    predicted_labels <- ifelse(boosting_preds > 0.5, 1, 0)
    
    boosting_confusion_matrix <- confusionMatrix(as.factor(predicted_labels), as.factor(actual_labels))
    accuracy_data <- c(accuracy_data, boosting_confusion_matrix$overall['Accuracy'])
    depth_data <- c(depth_data, j)
    nrounds_data <- c(nrounds_data, i)
    info_loss_data <- c(info_loss_data, boost1$evaluation_log$train_logloss[i])
  }
}

boosting_accuract_df <- data.frame(
  depth = depth_data,
  nrounds = nrounds_data, 
  info_loss = info_loss_data,
  accuracy = accuracy_data
)

boosting_accuract_df <- boosting_accuract_df[order(-boosting_accuract_df$accuracy),]
boosting_accuract_df









#max_depth = 2, nrounds = 100, accuracy = 0.8602620
#remove outliers
df_temp <- df
outliers <- boxplot(df_temp$RestingBP, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$RestingBP %in% outliers),]

outliers <- boxplot(df_temp$Cholesterol, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Cholesterol %in% outliers),]

outliers <- boxplot(df_temp$Oldpeak, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Oldpeak %in% outliers),]
##########train&test
class0 <- filter(df_temp, df_temp$HeartDisease == 1)
class0

class1 <- subset(df_temp, !(HeartDisease %in% class0$HeartDisease))
class1

sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train_temp <- rbind(train0, train1)
train_temp <- train_temp[sample(1:nrow(train_temp)), ] #shuffle
train_temp

test_temp <- rbind(test0, test1)
test_temp <- test_temp[sample(1:nrow(test_temp)), ] #shuffle
test_temp

train_temp$HeartDisease <- ifelse(train_temp$HeartDisease == 1, 0, 1)
test_temp$HeartDisease <- ifelse(test_temp$HeartDisease == 1, 0, 1)


max_depth_lst <- c(seq(1, 11, 1))
nrounds_lst <- c(seq(100, 300, 100))
accuracy_data <- c()
depth_data <- c()
nrounds_data <- c()
info_loss_data <- c()
for (i in nrounds_lst){
  for (j in max_depth_lst){
    boost1 <- xgboost(data = as.matrix(train_temp[, -which(names(train_temp) == "HeartDisease")]), 
                      label = train_temp$HeartDisease,  
                      max.depth = j,
                      eta = 1,
                      nthread = 5,
                      nrounds = i,
                      objective = "binary:logistic",
                      verbose = 0  # Set verbose to 0 to reduce log output, or use 1 for more verbose
    )
    test_matrix <- as.matrix(test_temp[, -which(names(test_temp) == "HeartDisease")])
    boosting_preds <- predict(boost1, test_matrix)
    actual_labels <- test_temp$HeartDisease
    predicted_labels <- ifelse(boosting_preds > 0.5, 1, 0)
    
    boosting_confusion_matrix <- confusionMatrix(as.factor(predicted_labels), as.factor(actual_labels))
    accuracy_data <- c(accuracy_data, boosting_confusion_matrix$overall['Accuracy'])
    depth_data <- c(depth_data, j)
    nrounds_data <- c(nrounds_data, i)
    info_loss_data <- c(info_loss_data, boost1$evaluation_log$train_logloss[i])
  }
}

boosting_accuract_df <- data.frame(
  depth = depth_data,
  nrounds = nrounds_data, 
  info_loss = info_loss_data,
  accuracy = accuracy_data
)

boosting_accuract_df <- boosting_accuract_df[order(-boosting_accuract_df$accuracy),]
boosting_accuract_df












#################################################################################
##KSVM
#################################################################################
setwd("C:/Users/39930/Desktop/ALY6040/project")
df <- read.csv("heart.csv")
df

df <- na.omit(df)
df
df$Sex <- as.factor(df$Sex)
df$ChestPainType <- as.factor(df$ChestPainType)
df$RestingECG <- as.factor(df$RestingECG)
df$ST_Slope <- as.factor(df$ST_Slope)
df$ExerciseAngina <- as.factor(df$ExerciseAngina)
df$HeartDisease <- as.factor(df$HeartDisease)

df[, c(4, 5, 8)] <- scale(select(df, c(RestingBP, Cholesterol, MaxHR)))
df

set.seed(1000)
class0 <- filter(df, df$HeartDisease == 0)
class0

class1 <- subset(df, !(HeartDisease %in% class0$HeartDisease))
class1

sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train <- rbind(train0, train1)
train <- train[sample(1:nrow(train)), ] #shuffle
train

test <- rbind(test0, test1)
test <- test[sample(1:nrow(test)), ] #shuffle
test

df_temp <- df
outliers <- boxplot(df_temp$RestingBP, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$RestingBP %in% outliers),]

outliers <- boxplot(df_temp$Cholesterol, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Cholesterol %in% outliers),]

outliers <- boxplot(df_temp$Oldpeak, plot=FALSE)$out
df_temp <- df_temp[-which(df_temp$Oldpeak %in% outliers),]
##########train&test
class0 <- filter(df_temp, df_temp$HeartDisease == 1)
class0

class1 <- subset(df_temp, !(HeartDisease %in% class0$HeartDisease))
class1

sample_split_class0 <- sample.split(Y = class0$HeartDisease, SplitRatio = .75)
train0 <- subset(class0, sample_split_class0 == TRUE)
test0 <- subset(class0, sample_split_class0 == FALSE)

sample_split_class1 <- sample.split(Y = class1$HeartDisease, SplitRatio = .75)
train1 <- subset(class1, sample_split_class1 == TRUE)
test1 <- subset(class1, sample_split_class1 == FALSE)

ratio_table <- data.frame(
  train = nrow(train0) / nrow(train1),
  test = nrow(test0) / nrow(test1),
  dataset = nrow(class0) / nrow(class1)
)
ratio_table

train_temp <- rbind(train0, train1)
train_temp <- train_temp[sample(1:nrow(train_temp)), ] #shuffle
train_temp

test_temp <- rbind(test0, test1)
test_temp <- test_temp[sample(1:nrow(test_temp)), ] #shuffle
test_temp



library(kernlab)
ksvm1 <- ksvm(HeartDisease~., data = train, kernel = 'rbfdot', prob.model = TRUE)
ksvm1
pred1 <- predict(ksvm1, test)
pred1

confusionMatrix(pred1, test$HeartDisease)

library(kernlab)
ksvm1 <- ksvm(HeartDisease~., data = train_temp, kernel = 'rbfdot', prob.model = TRUE)
ksvm1
pred1 <- predict(ksvm1, test_temp)
pred1

confusionMatrix(pred1, test_temp$HeartDisease)

#var of importance
initial_model <- ksvm(HeartDisease~., data = train, kernel = 'rbfdot', prob.model = TRUE)
initial_pred <- predict(initial_model, test)
confusion_bagging <- confusionMatrix(test$HeartDisease, initial_pred)
initial_accuracy <- confusion_bagging$overall['Accuracy']

feature_performance <- data.frame(Feature=names(train[1:11]), Accuracy=numeric(length=11))
for (i in 1:11) {
  #print(i)
  feature_subset <- train[, -i, drop=FALSE]
  feature_subset$HeartDisease <- train$HeartDisease
  
  temp_model <- ksvm(HeartDisease~., data = feature_subset, kernel = 'rbfdot', prob.model = TRUE)
  temp_pred <- predict(temp_model, test[,-i, drop = FALSE])
  confusion_bagging <- confusionMatrix(table(test$HeartDisease, temp_pred))
  temp_accuracy <- confusion_bagging$overall['Accuracy']
  
  feature_performance$Accuracy[i] <- temp_accuracy
}

feature_performance$AccuracyDifference = initial_accuracy - feature_performance$Accuracy
print(feature_performance[order(feature_performance$AccuracyDifference),])




###cross validation
train$HeartDisease <- ifelse(train$HeartDisease == 1, "Yes", "No")
test$HeartDisease <- ifelse(test$HeartDisease == 1, "Yes", "No")
df$HeartDisease <- ifelse(df$HeartDisease == 1, "Yes", "No")
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)
churn_svm_auc <- train(
  HeartDisease ~ ., 
  data = df,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)
print(churn_svm_auc)
churn_svm_auc$results

confusionMatrix(churn_svm_auc)



df_temp$HeartDisease <- ifelse(df_temp$HeartDisease == 1, "Yes", "No")
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)
churn_svm_auc <- train(
  HeartDisease ~ ., 
  data = df_temp,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)
print(churn_svm_auc)
confusionMatrix(churn_svm_auc)

#################################################################################
##Naive Bayes
#################################################################################
naive1 <- naiveBayes(HeartDisease~., data = train)
naive1
pred2 <- predict(naive1, test)
pred2

table2 <- table(test$HeartDisease, pred2)
table2

confusionMatrix(table2)



#var of importance
initial_model <- naiveBayes(HeartDisease~., data = train)
initial_pred <- predict(initial_model, test)
confusion_bagging <- confusionMatrix(table(test$HeartDisease, initial_pred))
initial_accuracy <- confusion_bagging$overall['Accuracy']

feature_performance <- data.frame(Feature=names(train[1:11]), Accuracy=numeric(length=11))
for (i in 1:11) {
  #print(i)
  feature_subset <- train[, -i, drop=FALSE]
  feature_subset$HeartDisease <- train$HeartDisease
  
  temp_model <- naiveBayes(HeartDisease~., data = feature_subset)
  temp_pred <- predict(temp_model, test[,-i, drop = FALSE])
  confusion_bagging <- confusionMatrix(table(test$HeartDisease, temp_pred))
  temp_accuracy <- confusion_bagging$overall['Accuracy']
  
  feature_performance$Accuracy[i] <- temp_accuracy
}

feature_performance$AccuracyDifference = initial_accuracy - feature_performance$Accuracy
print(feature_performance[order(feature_performance$AccuracyDifference),])











#remove outliers
naive1 <- naiveBayes(HeartDisease~., data = train_temp)
naive1
pred2 <- predict(naive1, test_temp)
pred2

table2 <- table(test_temp$HeartDisease, pred2)
table2

confusionMatrix(table2)

################
#var of importance
initial_model <- naiveBayes(HeartDisease~., data = train_temp)
initial_pred <- predict(initial_model, test_temp)
confusion_bagging <- confusionMatrix(table(test_temp$HeartDisease, initial_pred))
initial_accuracy <- confusion_bagging$overall['Accuracy']

feature_performance <- data.frame(Feature=names(train_temp[1:11]), Accuracy=numeric(length=11))
for (i in 1:11) {
  #print(i)
  feature_subset <- train_temp[, -i, drop=FALSE]
  feature_subset$HeartDisease <- train_temp$HeartDisease
  
  temp_model <- naiveBayes(HeartDisease~., data = feature_subset)
  temp_pred <- predict(temp_model, test_temp[,-i, drop = FALSE])
  confusion_bagging <- confusionMatrix(table(test_temp$HeartDisease, temp_pred))
  temp_accuracy <- confusion_bagging$overall['Accuracy']
  
  feature_performance$Accuracy[i] <- temp_accuracy
}

feature_performance$AccuracyDifference = initial_accuracy - feature_performance$Accuracy
print(feature_performance[order(feature_performance$AccuracyDifference),])




#cross validation
# Set up cross-validation
train_control <- trainControl(method = "cv", number = 10)
# Train the model using Naive Bayes
model <- train(HeartDisease~., data = df, method = "nb", trControl = train_control)
model

confusionMatrix(model)

train_control <- trainControl(method = "cv", number = 10)
# Train the model using Naive Bayes
model <- train(HeartDisease~., data = df_temp, method = "nb", trControl = train_control)
model
#################################################################################
##KNN
#################################################################################
library(class)
df$Sex <- as.numeric(as.factor(df$Sex))
df$ChestPainType <- as.numeric(as.factor(df$ChestPainType))
df$RestingECG <- as.numeric(as.factor(df$RestingECG))
df$ST_Slope <- as.numeric(as.factor(df$ST_Slope))
df$ExerciseAngina <- as.numeric(as.factor(df$ExerciseAngina))
df$HeartDisease <- as.numeric(as.factor(df$HeartDisease))

train$Sex <- as.numeric(as.factor(train$Sex))
train$ChestPainType <- as.numeric(as.factor(train$ChestPainType))
train$RestingECG <- as.numeric(as.factor(train$RestingECG))
train$ST_Slope <- as.numeric(as.factor(train$ST_Slope))
train$ExerciseAngina <- as.numeric(as.factor(train$ExerciseAngina))
train$HeartDisease <- as.numeric(as.factor(train$HeartDisease))

test$Sex <- as.numeric(as.factor(test$Sex))
test$ChestPainType <- as.numeric(as.factor(test$ChestPainType))
test$RestingECG <- as.numeric(as.factor(test$RestingECG))
test$ST_Slope <- as.numeric(as.factor(test$ST_Slope))
test$ExerciseAngina <- as.numeric(as.factor(test$ExerciseAngina))
test$HeartDisease <- as.numeric(as.factor(test$HeartDisease))

df_temp$Sex <- as.numeric(as.factor(df_temp$Sex))
df_temp$ChestPainType <- as.numeric(as.factor(df_temp$ChestPainType))
df_temp$RestingECG <- as.numeric(as.factor(df_temp$RestingECG))
df_temp$ST_Slope <- as.numeric(as.factor(df_temp$ST_Slope))
df_temp$ExerciseAngina <- as.numeric(as.factor(df_temp$ExerciseAngina))
df_temp$HeartDisease <- as.numeric(as.factor(df_temp$HeartDisease))

train_temp$Sex <- as.numeric(as.factor(train_temp$Sex))
train_temp$ChestPainType <- as.numeric(as.factor(train_temp$ChestPainType))
train_temp$RestingECG <- as.numeric(as.factor(train_temp$RestingECG))
train_temp$ST_Slope <- as.numeric(as.factor(train_temp$ST_Slope))
train_temp$ExerciseAngina <- as.numeric(as.factor(train_temp$ExerciseAngina))
train_temp$HeartDisease <- as.numeric(as.factor(train_temp$HeartDisease))

test_temp$Sex <- as.numeric(as.factor(test_temp$Sex))
test_temp$ChestPainType <- as.numeric(as.factor(test_temp$ChestPainType))
test_temp$RestingECG <- as.numeric(as.factor(test_temp$RestingECG))
test_temp$ST_Slope <- as.numeric(as.factor(test_temp$ST_Slope))
test_temp$ExerciseAngina <- as.numeric(as.factor(test_temp$ExerciseAngina))
test_temp$HeartDisease <- as.numeric(as.factor(test_temp$HeartDisease))

if (nrow(train) %% 2 == 0){
  k_val = seq(1,50,2)
  accuracy <- c()
  for (i in k_val){
    classifier_knn <- knn(train = train, 
                          test = test, 
                          cl = train$HeartDisease, 
                          k = i
    )
    table3 <- table(test$HeartDisease, classifier_knn)
    misClassError <- mean(classifier_knn != test$HeartDisease)
    accuracy <- append(accuracy, 1-misClassError)
  }
}
if (nrow(train) %% 2 == 1){
  k_val = seq(0,50,2)
  accuracy <- c()
  for (i in k_val){
    classifier_knn <- knn(train = train, 
                          test = test, 
                          cl = train$HeartDisease, 
                          k = i
    )
    table3 <- table(test$HeartDisease, classifier_knn)
    misClassError <- mean(classifier_knn != test$HeartDisease)
    accuracy <- append(accuracy, 1-misClassError)
  }
}
knn_df <- data.frame(
  K = k_val,
  Accuracy = accuracy
)
knn_df <- knn_df[order(-knn_df$Accuracy),]
knn_df

plot(knn_df$K, knn_df$Accuracy, type = "l")
print(paste("The best k is 3 with accuracy 0.8869"))



#var of importance
initial_model <- knn(train = train, 
                     test = test, 
                     cl = train$HeartDisease, 
                     k = 3)
#initial_pred <- predict(initial_model, test)
confusion_bagging <- confusionMatrix(table(test$HeartDisease, initial_model))
initial_accuracy <- confusion_bagging$overall['Accuracy']

feature_performance <- data.frame(Feature=names(train[1:11]), Accuracy=numeric(length=11))
for (i in 1:11) {
  #print(i)
  feature_subset <- train[, -i, drop=FALSE]
  feature_subset$HeartDisease <- train$HeartDisease
  
  temp_model <- knn(train = feature_subset, 
                    test = test[,-i, drop = FALSE], 
                    cl = train$HeartDisease, 
                    k = 3)
  #temp_pred <- predict(temp_model, test[,-i, drop = FALSE])
  confusion_bagging <- confusionMatrix(table(test$HeartDisease, temp_model))
  temp_accuracy <- confusion_bagging$overall['Accuracy']
  
  feature_performance$Accuracy[i] <- temp_accuracy
}

feature_performance$AccuracyDifference = initial_accuracy - feature_performance$Accuracy
print(feature_performance[order(feature_performance$AccuracyDifference),])












#cross validation
train_control <- trainControl(method = "cv", number = 10)

set.seed(123)  # for reproducibility
model_knn <- train(HeartDisease ~ ., data = df, method = "knn",
                   trControl = train_control,
                   preProcess = "scale",  # Scaling features
                   tuneLength = 10)       # Number of different k values to try

print(model_knn)


tune_grid <- expand.grid(k = c(1, 3, 5, 7, 9, 11))

# Train the KNN model using the specified tuning grid
knn_model <- train(HeartDisease ~ ., data = df, method = "knn",
                   trControl = trainControl(method = "cv", number = 10),
                   tuneGrid = tune_grid,
                   preProcess = "scale")
knn_model
confusionMatrix(knn_model)





if (nrow(train_temp) %% 2 == 0){
  k_val = seq(1,50,2)
  accuracy <- c()
  for (i in k_val){
    classifier_knn <- knn(train = train_temp, 
                          test = test_temp, 
                          cl = train_temp$HeartDisease, 
                          k = i
    )
    table3 <- table(test_temp$HeartDisease, classifier_knn)
    misClassError <- mean(classifier_knn != test_temp$HeartDisease)
    accuracy <- append(accuracy, 1-misClassError)
  }
}
if (nrow(train_temp) %% 2 == 1){
  k_val = seq(2,50,2)
  accuracy <- c()
  for (i in k_val){
    classifier_knn <- knn(train = train_temp, 
                          test = test_temp, 
                          cl = train_temp$HeartDisease, 
                          k = i
    )
    table3 <- table(test_temp$HeartDisease, classifier_knn)
    misClassError <- mean(classifier_knn != test_temp$HeartDisease)
    accuracy <- append(accuracy, 1-misClassError)
  }
}
knn_df <- data.frame(
  K = k_val,
  Accuracy = accuracy
)
knn_df <- knn_df[order(-knn_df$Accuracy),]
knn_df
