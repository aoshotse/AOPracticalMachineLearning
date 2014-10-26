setwd("C:/Users/BigAbe/Desktop/Work/MOOC/Practical Machine Learning/Course Project/AOPracticalMachineLearning")

#####################################
### Load Packages and Train Data ####
#####################################
library(caret)
library(ggplot2)
library(rattle)
library(ipred)
library(plyr)
library( GGally )
train1 <- read.csv("pml-training.csv")
thetest <- read.csv("pml-testing.csv") ## untouched until model is complete

#########################
### Clean Train Data ####
#########################

## Convert Variable Types
train1$user_name <- as.factor(train1$user_name)
train1$new_window <- as.factor(train1$new_window)
train1$cvtd_timestamp <- as.Date(train1$cvtd_timestamp, format="%m/%d/%Y %H:%M")
train1$classe <- as.factor(train1$classe)

## Function to Remove Variables with >= 50% Missng Values 
sumnans <- function(x){
	train2 <- 1:length(x[,1]) ## was train1
	j = 1
	namer <- vector()
	for (i in 1:length(colnames(x))){
		if (sum(is.na(x[,i])) <= 0.5*length(x[,1]) ) {
			j = j + 1
			train2 <- cbind(train2, x[,i])
			namer <- rbind(namer, colnames(x[i]))
			}
	}
	
	train2 <- train2[,-1]
	colnames(train2) <- namer
	return(train2)
}

train2 <- sumnans(train1)
train2 <- as.data.frame(train2)
# train2$user_name <- train1$user_name
# train2$new_window <- train1$new_window
# train2$cvtd_timestamp <- train1$cvtd_timestamp
train2$classe <- as.factor(train1$classe)
train2 <- na.omit(train2)

###########################################
### Re-Partition into Test0 and Train3 ####
###########################################
inTrain = createDataPartition(train2$classe, p = 0.85, list=FALSE)
train3 <- train2[inTrain,]
test0 <- train2[-inTrain,]
trainv <- train3[,-93]
testv <- test0[,-93]

############################
## Preprocessing with PCA ##
############################
## PCA pure train data set
trainz <- prcomp(trainv, tresh=0.9)
ztrain <- as.data.frame(trainz$x[,1:3])
ztrain$classe <- train3$classe

## PCA pure test data set
testz <- prcomp(testv, tresh=0.9)
ztest <- as.data.frame(testz$x[,1:3])
ztest$classe <- test0$classe


#########################
## Plotting Predictors ##
#########################
ggpairs(ztrain)
gg <- glm(I(as.numeric(classe)) ~., data=ztrain)
summary(gg)


#######################################
### 4-Fold Cross Validation Ztrain ####
#######################################
## Data set shuffled and split into 4 sets
## Model tested in 4 "folds"
## Out of sample error rate averaged bewteen them
train22 <- ztrain[sample(nrow(ztrain)),]
set1 <- train22[1:2795,]
set2 <- train22[2796:5590,]
set3 <- train22[5591:8385,]
set4 <- train22[8386:11180,]
sets <- list(set1, set2, set3, set4)
ooser <- c(0,0,0,0,0)
for (i in 1:length(sets)){
	x <- as.data.frame(sets[i])
	v <- sets[-i]
	cc <- as.data.frame(v[1])
	cc <- rbind(cc, as.data.frame(v[2]))
	cc <- rbind(cc, as.data.frame(v[3]))
	fit <- train(classe~.,method="rf", data=cc)
	pred2 <- predict(fit, newdata=x)
	ooser[i] <- 1 - confusionMatrix(x$classe, pred2)$overall[1]
	## print(ooser[i])
}


#####################################
## Test Model on in-Train Test Set ##
#####################################
fit1 <- train(classe~.,method="rf", data=ztrain)
pred11 <- predict(fit1, newdata=ztest)
ooser[5] <- 1 - confusionMatrix(ztest$classe, pred11)$overall[1]
er <- 100 * mean(ooser) ## expected Out of Sample error rate


#############################################
## Prepare Final Train Data for Predicting ##
#############################################
thetrain <- rbind(ztrain, ztest)


###############################################
## Preprocess Final Test Data for Predicting ##
###############################################
## thetest2 <- sumnans(thetest)
## thetest2 <- as.data.frame(thetest2)
## thetest2 <- na.omit(thetest2)
## Reduce test set to same variables as train set
thetest3 <- subset(thetest, select=(names(testv)))
## Convert data types
thetest3$user_name <- as.numeric(thetest3$user_name)
thetest3$new_window <- as.numeric(thetest3$new_window)
thetest3$cvtd_timestamp <- as.numeric(thetest3$cvtd_timestamp)
## Perform PCA
thetest3 <- prcomp(thetest2, tresh=0.9)
thetest3 <- as.data.frame(thetest3$x[,1:3])


######################
## Final Prediction ##
######################
fit2 <- train(classe~.,method="rf", data=thetrain)
pred22 <- predict(fit2, newdata=thetest3) ## Final predictions



