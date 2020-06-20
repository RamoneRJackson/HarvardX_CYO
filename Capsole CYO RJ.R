# Install all needed libraries if it is not present

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(dplyr)) install.packages("dplyr")
if(!require(caret)) install.packages("caret")
if(!require(e1071)) install.packages("e1071")
if(!require(class)) install.packages("class")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(PRROC)) install.packages("PRROC")
if(!require(reshape2)) install.packages("reshape2")
if(!require(funModeling)) install.packages("funModeling")
if(!require(corrplot)) install.packages("corrplot")
if(!require(dplyr)) install.packages("dplyr")
if(!require(gridExtra))install.packages("gridExtra")
# Loading all needed libraries

library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(ggplot2)
library(caret)
library(e1071)
library(class)
library(ROCR)
library(randomForest)
library(PRROC)
library(reshape2)
library(funModeling)
library(corrplot)
library(dplyr)

## Loading the dataset from my github profile
data <- read.csv("https://raw.githubusercontent.com/RamoneRJackson/HarvardX_CYO/master/datasets_228_482_diabetes.csv")

#####Evaluating the Data

#Finding the amount of observations and variables in the datset
dim(data)


#Viewing the datatypes of the variables in the dataset
str(data)

#Number of different ages in the dataset
n_distinct(data$Age)

#Checking to see the number of diabetes patients
data%>%group_by(Outcome)%>%summarise(count=n())


#Finding out if there any data with N/As
sapply(data, function(m){
  sum(is.na(m))
})

#Finding the top 6 ages represented in the dataset
data%>%group_by(Age)%>%summarise(count=n())%>%
  arrange(desc(count))%>%head(n=6)

#Top 6 ages of patients with diabetes
data%>%filter(Outcome==1)%>%group_by(Age)%>%summarise(count=n())%>%
  arrange(desc(count))%>%head(n=6)

#Viewing the first 6 rows of the dataset
data%>%head(n=6)



####Visualizaion 

#Age vs Density 
data%>%ggplot( aes(Age)) + 
  geom_histogram(aes(y = ..density..),binwidth = 1, colour = "black", fill = "white")+
  geom_density(alpha = 0.2, fill = "#FF6666") + ggtitle("Age vs Density")

##Most of the persons chosen for the dataset aged between 20 and 40

#Frequency between Outcomes

data %>%
  ggplot(aes(Outcome, fill= Outcome)) +
  theme_minimal()  +
  geom_bar() +labs(title = "Frequency between Outcomes",
       x = "Outcomes",
       y = "Frequency")

## There number of perss without the disease in the dataset almost doubles the amount of person wth the disease

# Plotting all numeric data
plot_num(data, bins=10)  


# Check the correlation between chemicals 
corrMatrix <- cor(data)
corrplot(corrMatrix, type = "upper", order = "hclust", 
         tl.col = "blue",tl.cex = 1, addrect = 8) 
##There is no high level of correlation. Therefore all columns should be kept 



#Modeling 

## Data Cleaning

#1. Setting outcome to "yes" or "no"

# Setting outcome as a character variable 
data<- data%>%mutate(Outcome= 
                       ifelse(Outcome ==1,"yes","no"))


#2. Setting outcome as a factor
data$Outcome <- as.factor(data$Outcome)

#Dataset after cleaning
head(data)

# Split the dataset into train and test set
# Set seed as a starting point

set.seed(1, sample.kind='Rounding')
train_index <- createDataPartition(y = data$Outcome, p = 0.25, list = F)

training_set <- data[-train_index,]
validation_set <- data[train_index,]

#Showing the dimension of the training set      
dim(training_set)   

#Showing the dimension of the validation set      
dim(validation_set)

# Set seed as a starting point
set.seed(1, sample.kind='Rounding')

# Defining the training control that will be used for the model

train.control <- trainControl(method = "cv", #Cross Validation method
                              classProbs = TRUE,
                              number = 5,  #The number of folds
                              summaryFunction = twoClassSummary)

#Function to determine the avg of the average of the matrics
Metrics_Avg <- function(f1_score, accuracy, sensitivity)
  {
     (f1_score + accuracy + sensitivity)/3 
    }

#Naive Guessing Model 
## Here we assume that if the Diabetes Pedigree Function is larger than 0.5, then
## the patient would have diabetes


# Set seed as a starting point
set.seed(1, sample.kind='Rounding')
predict_guess <- ifelse(validation_set$DiabetesPedigreeFunction >=
                          0.5,"yes","no")
predict_guess <- as.factor(predict_guess)

# Calculating the confusion matrix
confusion_matrix_guess <- confusionMatrix(predict_guess, validation_set$Outcome, positive = "yes")


#Storing the values of both the Accuracy and sensitivity level for model
Accuracy_guess <- confusion_matrix_guess$overall[['Accuracy']]
Sensitivity_guess <- sensitivity(predict_guess, validation_set$Outcome, positive="yes")


#Calculating F1 Score

recall <- sensitivity(predict_guess, validation_set$Outcome, positive="yes")
precision <- posPredValue(predict_guess, validation_set$Outcome, positive="yes")

F1_Score_guess <- (2 * recall* precision) / (precision + recall)

Metrics_Avg_guess <- Metrics_Avg(F1_Score_guess,Accuracy_guess, Sensitivity_guess )

results <- data.frame(Model_Name="Naive Guessing Model", Accuracy =Accuracy_guess,
                      F1_Score = F1_Score_guess,Sensitivity = Sensitivity_guess,
                      Matric_Avg = Metrics_Avg_guess)

#Printing the results of our Model
print(Accuracy_guess) 
print(F1_Score_guess)
print(Sensitivity_guess)



## Naive Bayes QDA Model

set.seed(1, sample.kind='Rounding')

fitting_qda<- train(Outcome ~., data = training_set , method = "qda",
                    metric = "ROC",
                    preProcess = c("scale", "center"),  # used to normalize the data
                    trControl= train.control)


# Predicting the patient disease status using the validation set
predict_qda<- predict( fitting_qda, validation_set)

# Calculating the confusion matrix
confusion_matrix_qda <- confusionMatrix(predict_qda, validation_set$Outcome, positive = "yes")


#Storing the Accuracy and sensitivity level for knn model
Accuracy_qda <- confusion_matrix_qda$overall[['Accuracy']]
Sensitivity_qda <- sensitivity(predict_qda, validation_set$Outcome, positive="yes")


#Calculating F1 Score

recall <- sensitivity(predict_qda, validation_set$Outcome, positive="yes")
precision <- posPredValue(predict_qda, validation_set$Outcome, positive="yes")

F1_Score_qda <- (2 * recall* precision) / (precision + recall)

Metrics_Avg_qda <- Metrics_Avg(F1_Score_qda,Accuracy_qda, Sensitivity_qda )

#Adding the result of this model to the results dataset
results <- results %>% add_row(Model_Name="Naive Bayes QDA Model",
                               Accuracy =Accuracy_qda,
                               F1_Score = F1_Score_qda,
                               Sensitivity = Sensitivity_qda,
                               Matric_Avg = Metrics_Avg_qda)


#Printing the results of our Model
print(Accuracy_qda) 
print(F1_Score_qda)
print(Sensitivity_qda)





##K-nearest neighbors Model

set.seed(1, sample.kind='Rounding')

fitting_knn<- train(Outcome ~., data = training_set , method = "knn",
                metric = "ROC",
                preProcess = c("scale", "center"),  # used to normalize the data
                trControl= train.control,
                tuneLength=10)


# Predicting the patient disease status using the validation set
predict_knn<- predict( fitting_knn, validation_set)

# Calculating the confusion matrix
confusion_matrix_knn <- confusionMatrix(predict_knn, validation_set$Outcome, positive = "yes")


#Storing the Accuracy and sensitivity level for knn model
Accuracy_knn <- confusion_matrix_knn$overall[['Accuracy']]
Sensitivity_knn <- sensitivity(predict_knn, validation_set$Outcome, positive="yes")


#Calculating F1 Score

recall <- sensitivity(predict_knn, validation_set$Outcome, positive="yes")
precision <- posPredValue(predict_knn, validation_set$Outcome, positive="yes")

F1_Score_knn <- (2 * recall* precision) / (precision + recall)


Metrics_Avg_knn <- Metrics_Avg(F1_Score_knn,Accuracy_knn, Sensitivity_knn )

#Adding the result of this model to the results dataset
results <- results %>% add_row(Model_Name="K-nearest neighbors Model",
                               Accuracy =Accuracy_knn,
                               F1_Score = F1_Score_knn,
                               Sensitivity = Sensitivity_knn,
                               Matric_Avg = Metrics_Avg_knn)


#Printing the results of our  Model
print(Accuracy_knn) 
print(F1_Score_knn)
print(Sensitivity_knn)


## Logistic Regression Model

set.seed(1, sample.kind='Rounding')

fitting_glm<- train(Outcome ~., data = training_set , method = "glm",
                metric = "ROC",
                preProcess = c("scale", "center"),  # used to normalize the data
                trControl= train.control,
                family = "binomial")

# Predicting the patient disease status using the validation set
predict_glm<- predict( fitting_glm, validation_set)

# Calculating the confusion matrix
confusion_matrix_glm <- confusionMatrix(predict_glm, validation_set$Outcome, positive = "yes")


#Storing the values of both the Accuracy and sensitivity level for lm model
Accuracy_glm <- confusion_matrix_glm$overall[['Accuracy']]
Sensitivity_glm <- sensitivity(predict_glm, validation_set$Outcome, positive="yes")


#Calculating F1 Score

recall <- sensitivity(predict_glm, validation_set$Outcome, positive="yes")
precision <- posPredValue(predict_glm, validation_set$Outcome, positive="yes")

F1_Score_glm <- (2 * recall* precision) / (precision + recall)


Metrics_Avg_glm <- Metrics_Avg(F1_Score_glm,Accuracy_glm, Sensitivity_glm )

#Adding the result of this model to the results dataset
results <- results %>% add_row(Model_Name="Logistic Regression Model",
                               Accuracy =Accuracy_glm,
                               F1_Score = F1_Score_glm,
                               Sensitivity = Sensitivity_glm,
                               Matric_Avg = Metrics_Avg_glm)

#Printing the results of our GLM Model
print(Accuracy_glm) 
print(F1_Score_glm)
print(Sensitivity_glm)


## Random Forrest Model

set.seed(1, sample.kind='Rounding')

fitting_rf<- train(Outcome ~., data = training_set , method = "rf",
                    metric = "ROC",
                    preProcess = c("scale", "center"),  # used to normalize the data
                    trControl= train.control, tuneLength=7)


# Predicting the patient disease status using the validation set
predict_rf<- predict( fitting_rf, validation_set)

# Calculating the confusion matrix
confusion_matrix_rf <- confusionMatrix(predict_rf, validation_set$Outcome, positive = "yes")


#Storing the Accuracy and sensitivity level for knn model
Accuracy_rf <- confusion_matrix_rf$overall[['Accuracy']]
Sensitivity_rf <- sensitivity(predict_rf, validation_set$Outcome, positive="yes")


#Calculating F1 Score

recall <- sensitivity(predict_rf, validation_set$Outcome, positive="yes")
precision <- posPredValue(predict_rf, validation_set$Outcome, positive="yes")

F1_Score_rf <- (2 * recall* precision) / (precision + recall)

Metrics_Avg_rf <- Metrics_Avg(F1_Score_rf,Accuracy_rf, Sensitivity_rf )

#Adding the result of this model to the results dataset
results <- results %>% add_row(Model_Name="Random Forrest Model",
                               Accuracy =Accuracy_rf,
                               F1_Score = F1_Score_rf,
                               Sensitivity = Sensitivity_rf,
                               Matric_Avg = Metrics_Avg_rf)


#Printing the results of our  Model
print(Accuracy_rf) 
print(F1_Score_rf)
print(Sensitivity_rf)




#Results from the Models
print(results)