library(xgboost)
library(Matrix)
library(AUC)
library(mice)
library(VIM)
library(ggplot2)
library(readr)
library(psych)
library(dplyr)
library(reshape2)
library(PerformanceAnalytics)
library(corrplot)
library(glmnet)
set.seed(2908)

santander_traindataset <- read.csv("C:/Users/vivek/Desktop/Marketing Project/train.csv")
santander_testdataset  <- read.csv("C:/Users/vivek/Desktop/Marketing Project/test.csv")

##### Remove the train IDs
santander_traindataset$ID <-NULL
##### Remove the test IDs
id <- santander_testdataset$ID
santander_testdataset$ID <-NULL
santander_traindataset$n0 <- apply(santander_traindataset, 1, FUN=function(x) {return( sum(x == 0) )})
santander_testdataset$n0 <- apply(santander_testdataset, 1, FUN=function(x) {return( sum(x == 0) )})

##### Remove all the constant features
for (f in names(santander_traindataset)) {
  if (length(unique(santander_traindataset[[f]])) == 1) {
    santander_traindataset[[f]] <- NULL
    santander_testdataset[[f]] <- NULL
  }
}

##### Remove identical features
combo <- combn(names(santander_traindataset), 2, simplify = F)
eli <- c()
for(i in combo) {
  feature1 <- i[1]
  feature2 <- i[2]
  
  if (!(feature1 %in% eli) & !(feature2 %in% eli)) {
    if (all(santander_traindataset[[feature1]] == santander_traindataset[[feature2]])) {
      eli <- c(eli, feature2)
    }
  }
}
feature <- setdiff(names(santander_traindataset), eli)
santander_traindataset <- santander_traindataset[, feature]
feature<-feature[-307]
santander_testdataset <- santander_testdataset[, feature]

#Reduce variables from santander_testdataset
for(f in colnames(santander_traindataset)[-307]){
  lim <- min(santander_traindataset[,f!="TARGET"])
  santander_testdataset[santander_testdataset[,f]<lim,f] <- lim
  lim <- max(santander_traindataset[,f!="TARGET"])
  santander_testdataset[santander_testdataset[,f]>lim,f] <- lim  
}
#Convert them to Matrix format
train<-as.matrix(santander_traindataset[,-307])
test<-as.matrix(santander_testdataset)
#######################################################
#PCA and Logistic Regression
######################################################
pca <- prcomp(santander_traindataset[,sapply(santander_traindataset,
       is.numeric)], center = TRUE, scale. = TRUE)
screeplot(pca, type="lines",col=3)
biplot(pca, scale = 0)
pca.pred <- predict(pca, test)
logreg<- glm(TARGET~.,data=santander_traindataset,family="binomial")


########################################################
#XGboost Model
########################################################
h <- sample(nrow(train),1000)
dval<-xgb.DMatrix(train[h,], label=santander_traindataset$TARGET[h],missing=0)
dtrain <- xgb.DMatrix(train[-h,],label=santander_traindataset$TARGET[-h],missing=0)
dtest <- xgb.DMatrix(test, missing=0)

watchlist <- list(val=dval, train=dtrain)

parameter <- list(  objective           = "binary:logistic", 
                    booster             = "gbtree",
                    eval_metric         = "auc",
                    eta                 = 0.25,
                    max_depth           = 7,
                    subsample           = 0.80,
                    colsample_bytree    = 0.95
)

c <- xgb.train(   params              = parameter, 
                  data                = dtrain, 
                  nrounds             = 100, 
                  verbose             = 1,
                  watchlist           = watchlist,
                  maximize            = TRUE
                  
)

summary(c)
trainpreds <- predict(c, train)
santander_traindataset$TARGET<-as.factor(santander_traindataset$TARGET)
#ROC CURVE
plot(roc(trainpreds,santander_traindataset$TARGET))
importance_matrix <- xgb.importance(feature, model = c)
#Plot Important matrix
xgb.plot.importance(importance_matrix)
################################################################################
