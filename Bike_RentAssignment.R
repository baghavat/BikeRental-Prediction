rm(list=ls())

getwd()

# We need to predict cnt feature in the given data set
# cnt: total number of bikes rented on any perticular day

#Load Libraries
x = c("ggplot2", "corrplot", "caret", "randomForest", "dplyr", "tidyr", "MLmetrics", "rpart", 'DataCombine')

#install.packages(x)
lapply(x, require, character.only = TRUE)

warnings()
data_Bike<-read.csv("day.csv", header=T)

#*********************************************************
# EXPLORATORY DATA ANALYSIS
#*********************************************************

# view first 6 rows of the test data
head(data_Bike)

colnames(data_Bike)

#Get dimesion of the data
dim(data_Bike)
# It has 731 rows and 16 columns

str(data_Bike)
# As we see we don't need to change any data type for any feature/column

# Observations:
# As we can see that temp, atemp, hum, windspeed, casual and registered are continous vfeatures and rest are categorical features.
# Target variable is continous.
# Implies, it is an regression problem.
summary(data_Bike)

#Histogram plot for continouss features

#histogram plot for temp
hist(data_Bike$temp)

#histogram plot for atemp
hist(data_Bike$atemp)

#histogram plot for atemp
hist(data_Bike$hum)

#histogram plot for atemp
hist(data_Bike$windspeed)

#histogram plot for atemp
hist(data_Bike$casual)

# analyze the distribution of  target variable 'cnt'
univariate_numeric(data_Bike$cnt)

#histogram plot for atemp
hist(data_Bike$registered)

#Observation: except casual and registered feature all other features are normally distributed and 
#             scale is between 0 and 1. We have to scale features casual and humidity later, before applying models.

# Scatter plots of  cntns variables w.r.t target variable
ggplot(data_Bike, aes(x= temp,y=cnt)) +
  geom_point()+
  geom_smooth()   # temp is not highly correlated to cnt

ggplot(data_Bike, aes(x= atemp,y=cnt)) +
  geom_point()+
  geom_smooth()  # atemp is not highly correlated to cnt
                 # But by looking at the plot of temp and atemp, we can clearly see that both follows same curve w.r.t cnt as y-axis
ggplot(data_Bike, aes(x= hum,y=cnt)) +
  geom_point()+
  geom_smooth()   # Correlation between cnt and hum is very low

ggplot(data_Bike, aes(x= windspeed,y=cnt)) +
  geom_point()+
  geom_smooth()   # Correlation between cnt and windspeed is very low


ggplot(data_Bike, aes(x= casual,y=cnt)) +
  geom_point()+
  geom_smooth()   # Correlation between cnt and casual is very low


ggplot(data_Bike, aes(x= registered,y=cnt)) +
  geom_point()+
  geom_smooth()   # this plot is showing linear behaviour, this feature can be used for univariate analysis.
                  # cnt and registered feature are very highly correlated, and seems linear regression would be a better model if we
                  #  go for univariate analysis.

#*************************************************************
# Missing Value Analysis
#*************************************************************

# Count total number of missing values

missing_value=data.frame(apply(data_Bike,2,function(input){sum(is.na(input))}))
missing_value$features=row.names(missing_value)
names(missing_value)[1]= "Missing_Percentage"
missing_value$Missing_Percentage=(missing_value$Missing_Percentage/nrow(data_Bike))
missing_value

# No missing values are present

#*************************************************************
# Outlier Analysis
#*************************************************************

# Checking which feature has outliers
ggplot(data = data_Bike, aes(x = "", y = temp)) + 
  geom_boxplot() # No outliers

ggplot(data = data_Bike, aes(x = "", y = atemp)) + 
  geom_boxplot() # No outliers

ggplot(data = data_Bike, aes(x = "", y = windspeed)) + 
  geom_boxplot()   # It has outliers

ggplot(data = data_Bike, aes(x = "", y = hum)) + 
  geom_boxplot() # Outliers are present

ggplot(data = data_Bike, aes(x = "", y = casual)) + 
  geom_boxplot() # Outliers are present

ggplot(data = data_Bike, aes(x = "", y = registered)) + 
  geom_boxplot() # No outliers

# x[!x %in% boxplot.stats(x)$out], by this formula we can directly remove the outliers from the desired feature.

cnames=colnames(data_Bike)

 #loop to remove outliers from all variables
 for(i in cnames){
   print(i)
   outliers = data_Bike[,i][data_Bike[,i] %in% boxplot.stats(data_Bike[,i])$out]
   print(length(outliers))
   data_Bike = data_Bike[which(!data_Bike[,i] %in% outliers),]
 }

#Outliers have been removed
ggplot(data = data_Bike, aes(x = "", y = casual)) + 
  geom_boxplot() # Outliers are present

dim(data_Bike)
data_Bike<-read.csv("day.csv", header=T)
# After removing outliers we are left out with 655 rows out of 731 rows. Which might be a problem, because of good amount
# of data loss


#****************************************************************
# Feature selection
#****************************************************************

# Correlation graph
data_New=subset(data_Bike,select = c("temp","atemp","windspeed","hum","casual","registered","cnt"))

#model_rf= randomForest(cnt ~ ., data = data_Bike, ntree = 100, keep.forest = FALSE, importance = TRUE)


cr=cor(data_New)
corrplot(cr,type = "lower")
# As we can observe that, our target variable is very less dependent on feature hum and windspeed, so we can remove both
# And also temp and atemp are highly correlated to each other, it's better to remove one of them. I will remove atemp.

# Removal of features
data_Bike = subset(data_Bike,select=-c(dteday,atemp,hum, windspeed))
dim(data_Bike)

colnames(data_Bike)

#****************************************************************
# Feature Normalization
#****************************************************************

# As we observed above, casual and registered feature requires normalization

cnames=c("casual","registered")

for(iIterCol in cnames){
  print(iIterCol)
    data_Bike[,iIterCol] = (data_Bike[,iIterCol] - min(data_Bike[,iIterCol]))/
    (max(data_Bike[,iIterCol] - min(data_Bike[,iIterCol])))
}

ggplot(data = data_Bike, aes(x = "", y = casual)) + 
  geom_boxplot() 
# As we can see casual and registered feature has been normalized

#***********************************************************************************
#  Sampling
#***********************************************************************************

#Clean the environment
rmExcept("data_Bike")

#Dividing the data into train and test using stratified sampling method.
set.seed(3)
id = sample(2,nrow(data_Bike), prob = c(0.7,0.3),replace=TRUE)
bike_train = data_Bike[id==1,]
bike_test  = data_Bike[id==2,]

dim(bike_train)
dim(bike_test)
# Data has been divided correctly

#**********************************************************************************
#   Developing Models
#**********************************************************************************

##### Linear Regression

model_lm=lm(cnt~ weathersit+temp+casual+registered,data = bike_train)
summary(model_lm)
# R squared and adjusted R-squared error is 1, it might be possible that our model is over fitting.
# p-value is 2.2e^-16, which is a good sign

na.omit(model_lm)
predic=predict(model_lm,bike_test)

plot(bike_test$cnt,type='l',col="green")
lines(predic,type="l",col="blue")

# Errors
RMSE(pred  = predic, obs = bike_test$cnt)
MAPE(predic,bike_test$cnt)
# Root mean square error is 4.627063e-12
# MAPE is 1.436967e-15
# Let's visualize actual and predicted values, both are over lapping. Implies prediction is without any error.
# Train error and test error both are almost 0.
# This might be the best model, as we have observed while visualizing as well that linear model might be the best fit model.


##### Decision Tree

#Training the model
model_dt=rpart(cnt~workingday+mnth+holiday+weekday +weathersit+temp+casual+registered,data = bike_train,control = list(minsplit = 10, maxdepth = 20, cp = 0.01), method="anova")
plot(model_dt,margin=0.1)
# We can tune our model by changing hyper parameters i.e. depth and splits
text(model_dt,use.n = TRUE,pretty=TRUE)#control = list(minsplit = 10, maxdepth = 10, cp = 0.01))

predic <- predict(model_dt, newdata = bike_test)

plot(bike_test$cnt,type='l',col="green")
lines(predic,type="l",col="blue")
# After plotting predicted values and actual values, we can see the error please.

RMSE(pred  = predic, obs = bike_test$cnt)
MAPE(predic,bike_test$cnt)
# Root mean square error is 485.812
# MAPE is 0.1345875

##### Random Forest

model_rf=randomForest(cnt~workingday+mnth+holiday+weekday +weathersit+temp+casual+registered,data = bike_train,control = list(minsplit = 10, maxdepth = 20, cp = 0.01), method="anova")
plot(model_rf,margin=0.1)
# We can tune our model by changing hyper parameters i.e. depth and splits

predic=predict(model_rf,newdata = bike_test,type="class")

# Root mean square error is  334.4366
plot(bike_test$cnt,type='l',col="green")
lines(predic,type="l",col="blue")

RMSE(pred  = predic, obs = bike_test$cnt)
MAPE(predic,bike_test$cnt)
# Root mean square error is  341.242
# MAPE is 0.09948815

# Observations:
# i. Linear regression alg is giving us the best result compare to other Algo's.
# ii. It was visible while exploring the data, since 'registered' feature was linearly dependent on target feature 'cnt'. 
#      It was highly observable, that Multiple linear regression is going to be our best fit algorthm.


#############################################SAMPLE DATA##########################################################

# I have created sample data with madeup values.
sample_data=read.csv("Sample_Data.csv",header = TRUE)

#Preprocessing sample input data
sample_data = subset(sample_data,select=-c(dteday,atemp,hum, windspeed))
dim(sample_data)
cnames=c("casual","registered")

for(iIterCol in cnames){
  print(iIterCol)
  sample_data[,iIterCol] = (sample_data[,iIterCol] - min(sample_data[,iIterCol]))/
    (max(sample_data[,iIterCol] - min(sample_data[,iIterCol])))
}

predic=predict(model_lm,sample_data)
View(data.frame( predic))

#This will create a sample output.csv file in set directory and store the outputs in sequence of input file.
write.csv(predic,"Sample_Output.csv", col.names = FALSE)
