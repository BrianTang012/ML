install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("corrplot")
install.packages("factoextra")
install.packages("NbClust")

library(dplyr)
library(ggplot2)
library(caret)
library(corrplot)
library(factoextra)
library(NbClust)
library(class)
#EDA 
wine <- read.table("winequality-white.csv", header = T, sep = ";")
summary(wine)

is.null(wine)
#FALSE result was obtained
summary(is.na(wine))

#No na values
str(wine)

wine<-wine%>%mutate(quality=as.factor(ifelse(quality>5, 1,0)))
normalise <-function(x) { (x -min(x))/(max(x)-min(x)) }
wine[,1:11] <- sapply(wine[,1:11], normalise)
wine

#Data visualisation
wine %>% ggplot(aes(x=quality, y= alcohol,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y=chlorides,fill = quality )) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= residual.sugar,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= sulphates,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= density,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= pH,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= free.sulfur.dioxide,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= total.sulfur.dioxide,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= citric.acid,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= fixed.acidity,fill = quality)) + geom_boxplot(show.legend=F)
wine %>% ggplot(aes(x=quality, y= volatile.acidity,fill = quality)) + geom_boxplot(show.legend=F)

#splitting dataset
set.seed(100)
training.idx <- sample(1: nrow(wine), size = nrow(wine)*0.8)
train.data <- wine[training.idx,]
test.data <- wine[-training.idx,]
#make correlation plot
corrplot(cor(train.data[,-12]), type = "upper", method = "color", 
         addCoef.col = "black", number.cex = 0.6)

#kNN classification
set.seed(101)
ac<-rep(0, 30)
for(i in 1:30){
  set.seed(101)
  knn.i<-knn(train.data[,1:11], test.data[,1:11], cl=train.data$quality, k=i)
  ac[i]<-mean(knn.i ==test.data$quality)
  cat("k=", i, " accuracy=", ac[i], "\n")
}
#Accuracy plot
plot(ac, type="b", xlab="K",ylab="Accuracy")
knn1<-knn(train.data[,1:11], test.data[,1:11], cl=train.data$quality, k=3) 
#k = 3 instead of k = 1 since k = 1 will induce the highest noise especially for large dataset, and k has to be within n = 12
table(knn1, test.data$quality)

mean(knn1 ==test.data$quality) #Accuracy: 0.7520408

#logistic regression
mlogit <- glm(quality ~., data = train.data, family = "binomial")
summary(mlogit)

#predicted probability P(Y=1)
Pred.p <-predict(mlogit, newdata =test.data, type = "response")
y_pred_num <-ifelse(Pred.p > 0.5, 1, 0)
y_pred <-factor(y_pred_num, levels=c(0, 1 ))
#Accuracy of the classification
table(y_pred,test.data$quality)

mean(y_pred ==test.data$quality) # Accuracy: 0.7316327


#Improved logistic(removal of outlier and dependent variable) 
mlogit1 <- glm(quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+
                 +               free.sulfur.dioxide+total.sulfur.dioxide+pH+
                 +               sulphates+alcohol, data = train.data, family = "binomial")
Pred.p1 <-predict(mlogit1, newdata =test.data, type = "response")
y_pred_num1 <-ifelse(Pred.p1 > 0.5, 1, 0)
y_pred1 <-factor(y_pred_num1, levels=c(0, 1))
table(y_pred1,test.data$quality)

mean(y_pred1 ==test.data$quality) # Accuracy: 0.7357143
#comparing between the 3 models
confusionMatrix(test.data$quality,knn1, positive='1')

confusionMatrix(test.data$quality,y_pred, positive='1')
confusionMatrix(test.data$quality,y_pred1,positive='1')

#Reimport dataframe for clustering
wine <- read.table("winequality-white.csv", header = T, sep = ";")
###USING K-means Clustering
# 1. EDA
# 1.1 number of rows and cols --> 4898 observations, 12 features
num_rows = nrow(wine)
num_cols = ncol(wine)
Num_rows #4898 rows
Num_cols #12 cols

#1.2 check for missing values --> return FALSE so no null values
is.null(wine) #FALSE

#2.Data Preparation
#2.1 Normalisation
normalize = function(x){
  return((x-min(x))/(max(x)-min(x)))
} 

wine$fixed.acidity <- normalize(wine$fixed.acidity)
wine$volatile.acidity <- normalize(wine$volatile.acidity)
wine$citric.acid <- normalize(wine$citric.acid)
wine$residual.sugar <- normalize(wine$residual.sugar)
wine$chlorides <- normalize(wine$chlorides)
wine$free.sulfur.dioxide <- normalize(wine$free.sulfur.dioxide)
wine$total.sulfur.dioxide <- normalize(wine$total.sulfur.dioxide)
wine$density <- normalize(wine$density)
wine$pH <- normalize(wine$pH)
wine$sulphates <- normalize(wine$sulphates)
wine$alcohol <- normalize(wine$alcohol)
wine$quality <- normalize(wine$quality)
wine
wine_scaled1 <- wine
wine_scaled1
class(wine_scaled1)

#3 kmeans clustering
#we now use the elbow plot method to determine the optimal number of clusters K
wcss <- function(k) {
  kmeans(wine_scaled1,k,nstart = 5)$tot.withinss
}
#this is the function used to compute the total within-cluster sum of square
k.values <- 1:15
set.seed(120)
#We are now computing and plotting wcss for k = 1 to k = 15

wcss_k <- sapply(k.values,wcss)
plot(k.values,wcss_k,type='b',pch=19,frame=FALSE,xlab="Number of clusters K",ylab="Total within-clusters sum of squares",main='Elbow Plot')
#we can see that our elbow plot suggests that 3 is the optimal number of clusters

set.seed(120)
samplewine <- sample(1:nrow(wine_scaled1),nrow(wine_scaled1)*0.5)
sample.wine <- wine_scaled1[samplewine,]
sample.wine #4898/2=2449 Observations and 12 columns
fviz_nbclust(sample.wine,kmeans,method='silhouette')
#This is to find the optimal number of clusters using the sihouette method. Sometimes the elbow plot is hard to view and using this method is easier.Silhouette method is not optimal for large datasets hence a sample of the data was taken instead to find k.
#We can see that both methods give the same optimal value of k=2.

#Let us now try another method using the NbClust() function to find the optimal number of clusters. This is because the elbow plot is ambiguous in showing the best k value. This code takes up to 3min to run due to the immense number of indices the program has to run through.
NbClust(sample.wine,diss=NULL,distance="euclidean",min.nc =2,max.nc=15,method="kmeans",index="all")
#This tells us that k=2

#Let us now do a final clustering of the data set using k=2
set.seed(120)
k2.final <- kmeans(wine_scaled1,2,nstart=5)
k2.final
#Output of above command
#K-means clustering with 2 clusters of sizes 2780, 2118

wine_scaled1 %>% mutate(Cluster = k2.final$cluster) %>% group_by(Cluster) %>% summarise_all("mean")
wine_scaled1 %>% mutate(Cluster = k2.final$cluster) %>% group_by(Cluster) %>% summarise_all("range")
wine_scaled1 %>% mutate(Cluster = k2.final$cluster) %>% group_by(Cluster) %>% count()
#This code line allows us to find the size of each cluster.

#Now let us plot the scatter plot to observe the trends in the two clusters
wine_scaled1$cluster = k2.final$cluster
ggplot(wine_scaled1,aes(alcohol,free.sulfur.dioxide,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting free.sulfur.dioxide against alcohol clusters")
ggplot(wine_scaled1,aes(alcohol,fixed.acidity,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting fixed.acidity against alcohol clusters")
ggplot(wine_scaled1,aes(alcohol,residual.sugar,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting residual.sugar against alcohol clusters")
ggplot(wine_scaled1,aes(pH,free.sulfur.dioxide,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting free.sulfur.dioxide against pH clusters")
ggplot(wine_scaled1,aes(pH,fixed.acidity,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting fixed.acidity against pH clusters")
ggplot(wine_scaled1,aes(pH,residual.sugar,colour=as.factor(cluster)))+geom_point() +labs(title="Plotting residual.sugar against pH clusters")

