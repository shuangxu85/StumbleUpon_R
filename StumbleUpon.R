# 1. Set Environment and load the data
library(mice)
library(readr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(corrplot) #correlogram
library(tiff)
library(GGally)
library(fastDummies)
library(nnet)
library(caret)
library(NeuralNetTools)
library(randomForest)
library(rpart)
library(gbm)
library(fields)
library(ROCit)

source("~/Documents/Babson-MBA/Spring - 2019/DE&A/Data/BabsonAnalytics.R")

df = read.csv('/Users/shuangxu/Documents/Babson-MBA/Spring - 2019/DE&A/Data/stumbleupon.csv')
str(df) # check if there are special characters 

# Reload data
df = read.csv('/Users/shuangxu/Documents/Babson-MBA/Spring - 2019/DE&A/Data/stumbleupon.csv', na.strings = c("?","1?"))
#~~ read "?" and "1?" as null valuse 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2.Data Exploration

## check data
str(df)
summary(df)
glimpse(df)

## delete some features 
df <- df[ ,-which(names(df) == c("url",'urlid','boilerplate'))] 
df$framebased = NULL
#~~ all value in "framebased" are zero

## List all variables
ls(df)


## check the shape of the data 
nrow(df)
length(df)
dim(df)

# Unique values per column
lapply(df, function(x) length(unique(x))) 

## 2.1 Missing values 

### Check which variables contain missing values 
sapply(df, function(x) {sum(is.na(x))})

### visualize the percentage of missing values for relevant features 
missing_values <- df %>% summarize_all(funs(sum(is.na(.))/n()))
missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>% 
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  coord_flip()+theme_bw()

### for label, replace null with 1 
df$label[is.na(df$label)] <- 1
table(df$label)

### for alchemy_category, replace null with unknown
### for alchemy_category_score, replace with 0.400001
df_ac_null <- df[which(is.na(df$alchemy_category)),] # show rows where alchemy_category is null
df_ukn <- df[which(df$alchemy_category=='unknown'),] # show rows where alchemy_category is unknown

df$alchemy_category[is.na(df$alchemy_category)] <- 'unknown'
df$alchemy_category_score[is.na(df$alchemy_category_score)] <- 0.400001

table(df$alchemy_category)

### for is_news, replace null with 0
df$is_news[is.na(df$is_news)] <- 0
table(df$is_news)

### for news_front_page, replace null with predicted value by using rf method 
idx_nfp = is.na(df$news_front_page)
impute_rf = mice(df,m=1,method="rf")
df = mice::complete(impute_rf)
df$news_front_page[idx_nfp] 

table(df$news_front_page)

## change feature type 
df <- df %>% mutate(
  is_news = factor(is_news),
  hasDomainLink = factor(hasDomainLink),
  lengthyLinkDomain = factor(lengthyLinkDomain),
  news_front_page = factor(news_front_page),
  label = factor(label)
)

summary(df)


## 2.2 Correlation and PCA

# correlation map 
df  %>%
  mutate_all(as.numeric) %>%
  select(everything()) %>%
  ggcorr(method = c("pairwise","spearman"), label = TRUE, angle = -0, hjust = 0.2) +
  coord_flip()


###Tried to apply Principle component analysis to subsets of attributes but it hurt our model's performance
##subset of linkquality features
#pca = prcomp(df[,c("html_ratio","avglinksize","commonlinkratio_1", "commonlinkratio_2","commonlinkratio_3","commonlinkratio_4")], scale=TRUE, center = TRUE)
#summary(pca)  # Choose the first 2 components to capture 68% of variance and signify link quality
#pred = predict(pca, df[,c("html_ratio","avglinksize","commonlinkratio_1", "commonlinkratio_2","commonlinkratio_3","commonlinkratio_4")])
#df$link_quality1 = pred[,1]    #captures 51% of the variance
#df$link_quality2 = pred[,2]    #captures 17% of the variance
#df$commonlinkratio_1 = NULL
#df$commonlinkratio_2 = NULL
#df$commonlinkratio_3 = NULL
#df$commonlinkratio_4 = NULL
#df$avglinksize = NULL
#df$numberOfLinks = NULL
#df$html_ratio = NULL

##Subset of content quality features
#pca = prcomp(df[,c("non_markup_alphanum_characters","spelling_errors_ratio","image_ratio")], scale=TRUE, center = TRUE)
#summary(pca)  # 3 out of 4 components capture enough variance - no point in PCA of these attributes

##Speed of page loading features
#pca = prcomp(df[,c("frameTagRatio","compression_ratio")], scale=TRUE, center = TRUE)
#summary(pca)  # 1st component captures 58% - no point in PCA of these attributes


## 2.3 Individual feature visualisations

### alchemy_category
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = reorder(alchemy_category,alchemy_category,function(x)-length(x)))) + 
  geom_bar(aes(fill = label)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## 2.4 Exploratory data analysis

## is_news vs. frameTagRatio vs.label 
ggplot(df, aes(x=frameTagRatio)) +
  geom_density(aes(fill = label), alpha = 0.5) +
  facet_wrap(~is_news)

### alchemy_category vs.label
df %>%
  ggplot() +
  geom_bar(aes(alchemy_category, fill = label), position = "dodge")  ## change scale??

df %>%
  group_by(alchemy_category, label) %>%
  count()

###News Features
## alchemy_category vs is_news vs.label
df[which(df$is_news=='1'),] %>%
  ggplot() + ylab("Number of news pages") +
  geom_bar(aes(alchemy_category, fill = label), position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## alchemy_category vs news_front_page vs.label
df[which(df$news_front_page=='1'),] %>%
  ggplot() + ylab("Number of frontpage news pages") +
  geom_bar(aes(alchemy_category, fill = label), position = "dodge") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

###Speed of Page loading features
## alchemy_category vs. compression ratio vs. label
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = alchemy_category, y = compression_ratio, fill=label )) +
  geom_boxplot() + scale_y_continuous(limits=c(0.35,0.6),breaks = scales::pretty_breaks(n = 10)) +
  theme_grey()+   theme(axis.text.x = element_text(angle = 90, hjust = 1))


## alchemy_category vs. frameTagRatio vs. label
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = alchemy_category, y = frameTagRatio, fill=label )) +
  geom_boxplot() + scale_y_continuous(limits=c(0.035,0.075),breaks = scales::pretty_breaks(n = 10)) + 
  theme_grey()+   theme(axis.text.x = element_text(angle = 90, hjust = 1))


###Page Content Quality features
## alchemy_category vs. non_markup_alphanum_characters vs. label
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = alchemy_category, y = non_markup_alphanum_characters, fill=label )) +
  geom_boxplot() + scale_y_continuous(limits=c(0,3500),breaks = scales::pretty_breaks(n = 10)) +
  theme_grey()+   theme(axis.text.x = element_text(angle = 90, hjust = 1))

## alchemy_category vs. image_ratio vs. label
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = alchemy_category, y = image_ratio, fill=label )) +
  geom_boxplot() + scale_y_continuous(limits=c(-1,0.2),breaks = scales::pretty_breaks(n = 10)) +
  theme_grey()+   theme(axis.text.x = element_text(angle = 90, hjust = 1))

## alchemy_category vs. spelling_error vs. label
ggplot(df[which(df$alchemy_category!='unknown'),], aes(x = alchemy_category, y = spelling_errors_ratio, fill=label )) +
  geom_boxplot() + scale_y_continuous(limits=c(0.04,0.2),breaks = scales::pretty_breaks(n = 10)) +
  theme_grey()+   theme(axis.text.x = element_text(angle = 90, hjust = 1))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3.Feature Engineering to improve model performance

## 3.1 Create dummy variables for 'alchemy_category', 'hasDomainLink','is_news','lengthyLinkDomain','news_front_page'
glimpse(df)
dummy_features <- c('alchemy_category', 'hasDomainLink','is_news','lengthyLinkDomain','news_front_page')
df <- fastDummies::dummy_cols(df, select_columns = dummy_features)
summary(df)

## delete original columns
df$alchemy_category = NULL
df$is_news = NULL
df$hasDomainLink = NULL
df$lengthyLinkDomain = NULL
df$news_front_page = NULL


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Modeling 

levels(df$label) <- c("Ephemeral", "Evergreen") # change the factor name in target columns 

## 4.1 split the 80% train and 20% test
trainingCases = sample(nrow(df),round(0.8*nrow(df)))
train = df[trainingCases,]
test = df[-trainingCases,]

## 4.2 Build Prediction Models 

set.seed(2019)
target <- names(df) %in% c("label") 

# 4.2.1 Random Forest model 
### with 10 cross validation and roc
rf <- train(x= train[!target], 
            y=train$label,
            data=train,
            metric='ROC',
            preProcess=c('center','scale'),
            method='rf', 
            trControl=trainControl(method="cv", number=10, classProbs = TRUE,
                                   summaryFunction = twoClassSummary))
rf$results
ggplot(rf)

### extracting variable importance and make graph with ggplot
rf_imp <- varImp(rf, scale = FALSE)
rf_imp <- rf_imp$importance
rf_gini <- data.frame(Variables = row.names(rf_imp), MeanDecreaseGini = rf_imp$Overall)

ggplot(rf_gini, aes(x=reorder(Variables, MeanDecreaseGini), y=MeanDecreaseGini, fill=MeanDecreaseGini)) +
  geom_bar(stat='identity') + coord_flip() + theme(legend.position="none") + labs(x="") +
  ggtitle('Variable Importance Random Forest') + theme(plot.title = element_text(hjust = 0.5))

#using the model to make predictions on the test set
pred_rf <- predict(rf, test)
cm_rf <- confusionMatrix(pred_rf, test$label, positive = "Evergreen")
plot(cm_rf$table)
cm_rf

###For the following non-tree algorithms, we need to make the model less complex by dropping irrelavent features 
###but this hurt our model's performance
#train$embed_ratio = NULL
#train$linkwordscore = NULL
#train$numberOfLinks = NULL
#train$numwords_in_url = NULL
#train$parametrizedLinkRatio = NULL
#train$is_news_1 = NULL
#train$is_news_0 = NULL
#train$lengthyLinkDomain_0 = NULL
#train$lengthyLinkDomain_1 = NULL
#train$news_front_page_0 = NULL
#train$news_front_page_1 = NULL
#train$hasDomainLink_0 = NULL
#train$hasDomainLink_1 = NULL

#test$embed_ratio = NULL
#test$linkwordscore = NULL
#test$numberOfLinks = NULL
#test$numwords_in_url = NULL
#test$parametrizedLinkRatio = NULL
#test$is_news_1 = NULL
#test$is_news_0 = NULL
#test$lengthyLinkDomain_0 = NULL
#test$lengthyLinkDomain_1 = NULL
#test$news_front_page_0 = NULL
#test$news_front_page_1 = NULL
#test$hasDomainLink_0 = NULL
#test$hasDomainLink_1 = NULL

## 4.2.2 Neural Network
nnet <- train(label~.,
              data=train,
              metric='ROC',
              preProcess=c('center','scale'),
              method='nnet',
              trace=FALSE,
              tuneLength=10,
              trControl=trainControl(method="cv", number=10, classProbs = TRUE,
                                     summaryFunction = twoClassSummary))
pred_nnet <- predict(nnet, test)
cm_nnet <- confusionMatrix(pred_nnet, test$label, positive = "Evergreen")
cm_nnet



## 4.2.3 Naive Bayes 
nb <- train(label~.,
            data=train,
            metric='ROC',
            preProcess=c('center','scale'),
            method='nb', 
            trace=FALSE,
            trControl=trainControl(method="cv", number=10, classProbs = TRUE,
                                   summaryFunction = twoClassSummary))
pred_nb <- predict(nb, test)
cm_nb <- confusionMatrix(pred_nb, test$label, positive = "Evergreen")
cm_nb

## 4.2.4 Logistic Regression

glm <- train(label~.,
             data=train,
             metric='ROC',
             preProcess=c('center','scale'),
             method='glm',
             family = 'binomial',
             trControl=trainControl(method="cv", number=10, classProbs = TRUE,
                                    summaryFunction = twoClassSummary))
pred_glm <- predict(glm, test)
cm_glm <- confusionMatrix(pred_glm, test$label, positive = "Evergreen")
cm_glm


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 5. Model Evaluation 

## ROC comparison chart for models 
model_list <- list(RF=rf,  NN = nnet, NB=nb, GLM = glm)
resamples <- resamples(model_list)
bwplot(resamples, metric="ROC")

## Metrics comparison for models 
cm_list <- list(RF=cm_rf, NN = cm_nnet, NB = cm_nb, GLM = cm_glm)
cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results

cm_results_max <- apply(cm_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
