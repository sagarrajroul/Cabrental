#loading libraries
libraries = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats')
lapply(libraries,require, character.only = TRUE)

#loading the datasets
train_cab = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test_cab = read.csv("test.csv")
test_pickup_time=test_cab['pickup_datetime']

##structure of the data
str(train_cab)
str(test_cab)

##exploratory data analysis
#changing variable types
train_cab$fare_amount = as.numeric(as.character(train_cab$fare_amount))
train_cab$passenger_count=round(train_cab$passenger_count)

#cleaning data
#as we know fare amount can't be -ve or 0 we have to remove some -ve or 0 values.
train_cab[which(train_cab$fare_amount < 1 ),]
nrow(train_cab[which(train_cab$fare_amount < 1 ),])
train_cab = train_cab[-which(train_cab$fare_amount < 1 ),]
#Passenger_count variable
for (i in seq(4,11,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train_cab[which(train_cab$passenger_count > i ),])))
}
train_cab[which(train_cab$passenger_count > 6 ),]
# Also we need to see if there are any passenger_count==0
train_cab[which(train_cab$passenger_count <1 ),]
nrow(train_cab[which(train_cab$passenger_count <1 ),])
# We will remove these 58 observations and 20 observation which are above 6 value because a cab cannot hold these number of passengers.
train_cab = train_cab[-which(train_cab$passenger_count < 1 ),]
train_cab = train_cab[-which(train_cab$passenger_count > 6),]
#Latitudes range from -90 to 90.Longitudes range from -180 to 180.Removing which does not satisfy these ranges
print(paste('pickup_longitude above 180=',nrow(train_cab[which(train_cab$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train_cab[which(train_cab$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train_cab[which(train_cab$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train_cab[which(train_cab$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train_cab[which(train_cab$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train_cab[which(train_cab$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train_cab[which(train_cab$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train_cab[which(train_cab$dropoff_latitude > 90 ),])))
# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.
# Also we will see if there are any values equal to 0.
nrow(train_cab[which(train_cab$pickup_longitude == 0 ),])
nrow(train_cab[which(train_cab$pickup_latitude == 0 ),])
nrow(train_cab[which(train_cab$dropoff_longitude == 0 ),])
nrow(train_cab[which(train_cab$pickup_latitude == 0 ),])
# there are values which are equal to 0. we will remove them.
train_cab = train_cab[-which(train_cab$pickup_latitude > 90),]
train_cab = train_cab[-which(train_cab$pickup_longitude == 0),]
train_cab = train_cab[-which(train_cab$dropoff_longitude == 0),]

# Make a copy
dataset=train_cab

###MISSING value ANALYSIS#####
missing_val = data.frame(apply(train_cab,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train_cab)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
missing_val

unique(train_cab$passenger_count)
unique(test_cab$passenger_count)
train_cab[,'passenger_count'] = factor(train_cab[,'passenger_count'], labels=(1:6))
test_cab[,'passenger_count'] = factor(test_cab[,'passenger_count'], labels=(1:6))

#FOR PASSENGER COUNT
train_cab$passenger_count[1000]
train_cab$passenger_count[1000] = NA
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
# Mode Method
getmode(train_cab$passenger_count)
#for fare_amount:
sapply(train_cab, sd, na.rm = TRUE)

train_cab$fare_amount[1000]
train_cab$fare_amount[1000]= NA
# Mean Method
mean(train_cab$fare_amount, na.rm = T)

#Median Method
median(train_cab$fare_amount, na.rm = T)

# kNN Imputation
train_cab = knnImputation(train_cab, k = 181)
train_cab$fare_amount[1000]
train_cab$passenger_count[1000]
sapply(train_cab, sd, na.rm = TRUE)

sum(is.na(train_cab))
str(train_cab)
summary(train_cab)

dataset1=train_cab

###OUTLAIR ANALYSIS####
# Boxplot for fare_amount
pl1 = ggplot(train_cab,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
vals = train_cab[,"fare_amount"] %in% boxplot.stats(train_cab[,"fare_amount"])$out
train_cab[which(vals),"fare_amount"] = NA

#lets check the NA's
sum(is.na(train_cab$fare_amount))

#Imputing with KNN
train_cab = knnImputation(train_cab,k=3)

# lets check the missing values
sum(is.na(train_cab$fare_amount))
str(train_cab)

dATASET2=train_cab

####FEATURE ENGINEERING####
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
train_cab$pickup_date = as.Date(as.character(train_cab$pickup_datetime))
train_cab$pickup_weekday = as.factor(format(train_cab$pickup_date,"%u"))# Monday = 1
train_cab$pickup_mnth = as.factor(format(train_cab$pickup_date,"%m"))
train_cab$pickup_yr = as.factor(format(train_cab$pickup_date,"%Y"))
pickup_time = strptime(train_cab$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train_cab$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test_cab$pickup_date = as.Date(as.character(test_cab$pickup_datetime))
test_cab$pickup_weekday = as.factor(format(test_cab$pickup_date,"%u"))# Monday = 1
test_cab$pickup_mnth = as.factor(format(test_cab$pickup_date,"%m"))
test_cab$pickup_yr = as.factor(format(test_cab$pickup_date,"%Y"))
pickup_time = strptime(test_cab$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test_cab$pickup_hour = as.factor(format(pickup_time,"%H"))

sum(is.na(train_cab))# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
train_cab = na.omit(train_cab) # we will remove that 1 row of na's

train_cab = subset(train_cab,select = -c(pickup_datetime,pickup_date))
test_cab = subset(test_cab,select = -c(pickup_datetime,pickup_date))

# 2.Calculate the distance travelled using longitude and latitude
dis_to_rad = function(dis){
  (dis * pi) / 180
}

haversine = function(long1,lat1,long2,lat2){
  #long1rad = dis_to_rad(long1)
  phi1 = dis_to_rad(lat1)
  #long2rad = dis_to_rad(long2)
  phi2 = dis_to_rad(lat2)
  delphi = dis_to_rad(lat2 - lat1)
  dellamda = dis_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}
# Using haversine formula to calculate distance fr both train and test
train_cab$dist = haversine(train_cab$pickup_longitude,train_cab$pickup_latitude,train_cab$dropoff_longitude,train_cab$dropoff_latitude)
test_cab$dist = haversine(test_cab$pickup_longitude,test_cab$pickup_latitude,test_cab$dropoff_longitude,test_cab$dropoff_latitude)

# We will remove the variables which were used to feature engineer new variables
train_cab = subset(train_cab,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test_cab = subset(test_cab,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

str(train_cab)
summary(train_cab)

# pickup_weekdat has p value greater than 0.05 
train_cab = subset(train_cab,select=-pickup_weekday)

#remove from test set
test_cab = subset(test_cab,select=-pickup_weekday)

####FEATURE SCALLING####
#Normality check
# qqnorm(train$fare_amount)
# histogram(train$fare_amount)
library(car)
# dev.off()
par(mfrow=c(1,2))
qqPlot(train_cab$fare_amount)                             # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train_cab$fare_amount)                           # truehist() scales the counts to give an estimate of the probability density.
lines(density(train_cab$fare_amount))  # Right skewed      # lines() and density() functions to overlay a density plot on histogram

#Normalisation

print('dist')
train_cab[,'dist'] = (train_cab[,'dist'] - min(train_cab[,'dist']))/
  (max(train_cab[,'dist'] - min(train_cab[,'dist'])))

# #check multicollearity
# library(usdm)
# vif(train_cab[,-1])
# 
# vifcor(train_cab[,-1], th = 0.9)

##### Splitting the Dataset ####
set.seed(1000)
tr.idx = createDataPartition(train_cab$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data = train_cab[tr.idx,]
test_data = train_cab[-tr.idx,]

rmExcept(c("test_cab","train_cab","dataset",'dataset1','dATASET2','df3','test_data','train_data','test_pickup_datetime'))

#####MODEL SELECTION####

##Linear Regression##
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)
str(train_data)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,2:6])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],lm_predictions)

##Decission Tree##
Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:6])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
#mae        mse       rmse        mape
#1.7591118  5.692008  2.3862525   0.2233173

##RANDOMFOREST REGRESSION##
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:6])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
#mae       mse      rmse      mape 
# 1.7464382 5.5594600 2.3578507 0.2298337 

##Improving Accurecy by using XGBOOST##
train_data_matrix = as.matrix(sapply(train_data[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data_data_matrix)

qplot(x = test_data[,1], y = xgb_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],xgb_predictions)
#mae       mse      rmse      mape 
#1.4687723 4.4483398 2.1091088 0.1840763 

##Finalizing and Saving model##
# In this step we will train our model on whole training Dataset and save that model for later use
train_data_matrix2 = as.matrix(sapply(train_cab[-1],as.numeric))
test_data_matrix2 = as.matrix(sapply(test_cab,as.numeric))

xgboost_model2 = xgboost(data = train_data_matrix2,label = train_cab$fare_amount,nrounds = 15,verbose = FALSE)

# Saving the trained model
saveRDS(xgboost_model2, "./final_Xgboost_model_using_R.rds")

# loading the saved model
super_model <- readRDS("./final_Xgboost_model_using_R.rds")
print(super_model)

# Lets now predict on test dataset
xgb = predict(super_model,test_data_matrix2)

xgb_pred = data.frame(test_pickup_time,"predictions" = xgb)

# Now lets write(save) the predicted fare_amount in disk as .csv format 
write.csv(xgb_pred,"xgb_predictionsR.csv",row.names = FALSE)