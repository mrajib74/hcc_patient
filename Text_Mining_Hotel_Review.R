library(NLP)
library(readtext)
library(tm)
library(SnowballC)
library(RColorBrewer)
library(wordcloud)
library(textstem)
library(data.table)
library(ggplot2)
library(RWeka)
library(dplyr)
library(tidyverse)
library(tidytext)
library(topicmodels)
library(lda)
library(lsa)
library(cluster)
options(mc.cores=1)

set.seed(1234)
setwd("D:\\Rajib\\XLRI\\Textmining\\Assignment")
hotelreview<-read.csv("hotelreviewcsv.csv",header=TRUE,sep=",")

####################################################
## Remove specialcharcters & newline from the revieW
####################################################
hotelreview$Review<-gsub("[\r\n]", "", hotelreview$Review)

####################################################
## expand contractions in an English-language source
####################################################

hotelreview$Review <- gsub("won't", "will not", hotelreview$Review)
hotelreview$Review <- gsub("can't", "can not", hotelreview$Review)
hotelreview$Review <- gsub("n't", " not", hotelreview$Review)
hotelreview$Review <- gsub("'ll", " will", hotelreview$Review)
hotelreview$Review <- gsub("'re", " are", hotelreview$Review)
hotelreview$Review <- gsub("'ve", " have", hotelreview$Review)
hotelreview$Review <- gsub("'m", " am", hotelreview$Review)
hotelreview$Review <- gsub("'d", " would", hotelreview$Review)
hotelreview$Review <- gsub("'s", "", hotelreview$Review)

##Lemmaization ofthe string
###############################################
hotelreview$Review  <- lapply(hotelreview$Review , lemmatize_strings)
itemsperlist <- vapply(hotelreview$Review , length, 1L)          ## How many items per list element
hotelreview <- hotelreview[rep(rownames(hotelreview), itemsperlist), ]   ## Expand the data frame
hotelreview$Review  <- unlist(hotelreview$Review , use.names = FALSE)  ## Replace with raw values
reviewsCorpus<-VCorpus(VectorSource(hotelreview$Review))

doclength <- length(reviewsCorpus)

# ignore extremely rare words i.e. terms that appear in less then 1% of the documents
minTermFreq <- doclength * 0.01
# ignore overly common words i.e. terms that appear in more than 50% of the documents
maxTermFreq <- doclength * .5

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
hotelReviewCrpous <- tm_map(reviewsCorpus, toSpace, "/")
hotelReviewCrpous <- tm_map(reviewsCorpus, toSpace, "@")
hotelReviewCrpous <- tm_map(reviewsCorpus, toSpace, "\\|")
hotelReviewCrpous<-tm_map(reviewsCorpus,removeWords,stopwords("english"))
hotelReviewCrpous<-tm_map(hotelReviewCrpous,removePunctuation,preserve_intra_word_dashes = TRUE)
hotelReviewCrpous<-tm_map(hotelReviewCrpous,removeNumbers)
hotelReviewCrpous<-tm_map(hotelReviewCrpous,stripWhitespace)
hotelReviewCrpous<-tm_map(hotelReviewCrpous,content_transformer(tolower))
#Remove additional stopwords
hotelReviewCrpous<-tm_map(hotelReviewCrpous,removeWords,c("can","the","you","which","they","got","via"
                                                          ,"will","bit","but","have","this","not"
                                                          ,"has","though","any","with","for","yet"
                                                          ,"than","when","get","and","was","were","i"))


tdmhotelReviewCrpous<-TermDocumentMatrix(hotelReviewCrpous,
                                control = list(bounds = list(global = c(minTermFreq, maxTermFreq) ))                                  )
mahotelReviewCrpous<-as.matrix(tdmhotelReviewCrpous)
freqhotelReviewCrpous<-sort(rowSums(mahotelReviewCrpous),decreasing = TRUE)
dataframehotelreview <- data.frame(word = names(freqhotelReviewCrpous),freq=freqhotelReviewCrpous)
plot(0:1, 0:1, type = "n", axes = FALSE, ann = FALSE)
wordcloud(dataframehotelreview$word, dataframehotelreview$freq,
          min.freq=25,
          max.words=300,
         colors=brewer.pal(8,"Dark2"),
        random.order=FALSE, rot.per=0.1)
title(main = "Most common word in Customer Review",
      font.main = 1, col.main = "thistle", cex.main = 1.5)




##############################################################################
# K mean clustering
#########################################################################3
dtmKmean<-DocumentTermMatrix(hotelReviewCrpous)
weightDtmTfidf<-weightTfIdf(dtmKmean)

mtrixweightDtm<-as.matrix(weightDtmTfidf)


f1<-function(x)
{
  return(sum(x^2)^.5)
}
eucliddistance<-function(x)
{
  mtrixweightDtm/apply(x,1,f1)
}
kk<-eucliddistance(mtrixweightDtm)
#accumulator for  results
dataframeacc <- data.frame()

#run kmeans for all clusters up to 100
for(i in 1:100){
  #Run kmeans for each level of i, allowing up to 100 iterations for convergence
  kmeans<- kmeans(x=weightDtmTfidf, centers=i, iter.max=100)
  
  #Combine cluster number and cost together, write to df
  dataframeacc<- rbind(dataframeacc, cbind(i, kmeans$tot.withinss))
  
}
names(dataframeacc) <- c("cluster", "cost")

#Calculate lm's for emphasis
lm(dataframeacc$cost[1:10] ~ dataframeacc$cluster[1:10])
lm(dataframeacc$cost[10:19] ~ dataframeacc$cluster[10:19])
lm(dataframeacc$cost[20:100] ~ dataframeacc$cluster[20:100])

dataframeacc$fitted <- ifelse(dataframeacc$cluster <10, (19019.9 - 550.9*dataframeacc$cluster), 
                         ifelse(dataframeacc$cluster <20, (15251.5 - 116.5*dataframeacc$cluster),
                                (13246.1 - 35.9*dataframeacc$cluster)))

#Elbow plot
ggplot(data=dataframeacc, aes(x=cluster, y=cost, group=1)) + 
  theme_bw(base_family="Garamond") + 
  geom_line(colour = "darkgreen") +
  theme(text = element_text(size=20)) +
  ggtitle("Elbow Plot\n") +
  xlab("\nClusters") + 
  ylab("Within-Cluster Sum of Squares\n") +
  scale_x_continuous(breaks=seq(from=0, to=100, by= 10)) +
  geom_line(aes(y= fitted), linetype=2)
set.seed(1234)
km<-kmeans(kk,10,nstart=20)

km$size
dataframe <- data.frame(text=sapply(hotelReviewCrpous, as.character),stringsAsFactors=F,row.names = NULL)
ak<-as.data.frame(as.matrix(weightDtmTfidf),km$cluster)
km$cluster


lsaspace<-lsa(t(mtrixweightDtm),dimcalc_share(share=0.8))
lsaspacetk<-as.data.frame(lsaspace$tk)
lsaspacedk<-as.data.frame(lsaspace$dk)
lsaspacesk<-as.data.frame(lsaspace$sk)

kmcluster<-kmeans(scale(lsaspacedk),centers=10,nstart=20)
kmcluster$size
c3_m_dk<-aggregate(cbind(V1,V2,V3)~kmcluster$cluster,data=lsaspacedk,FUN=mean)

clusplot(lsaspacedk, kmcluster$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

v=sort(colSums(mtrixweightDtm),decreasing=T)
wordFreq=data.frame(words=names(v),freq=v)
clust1<-wordFreq[kmcluster$cluster==1,]
clust2<-wordFreq[kmcluster$cluster==2,]
clust3<-wordFreq[kmcluster$cluster==3,]
clust4<-wordFreq[kmcluster$cluster==4,]
clust5<-wordFreq[kmcluster$cluster==5,]
clust6<-wordFreq[kmcluster$cluster==6,]
clust7<-wordFreq[kmcluster$cluster==7,]
clust8<-wordFreq[kmcluster$cluster==8,]
clust9<-wordFreq[kmcluster$cluster==9,]
clust10<-wordFreq[kmcluster$cluster==10,]
###################################################################
##Word cloud for each of the 10 clusters
##################################################################
wordcloud(clust1$words,clust1$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 1 Wordcloud", col.main = "grey14")
wordcloud(clust2$words,clust2$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 2 Wordcloud", col.main = "grey14")
wordcloud(clust3$words,clust3$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 3 Wordcloud", col.main = "grey14")
wordcloud(clust4$words,clust4$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 4 Wordcloud", col.main = "grey14")
wordcloud(clust5$words,clust5$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 5 Wordcloud", col.main = "grey14")
wordcloud(clust6$words,clust6$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 6 Wordcloud", col.main = "grey14")
wordcloud(clust7$words,clust7$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster7 Wordcloud", col.main = "grey14")
wordcloud(clust8$words,clust8$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 8 Wordcloud", col.main = "grey14")
wordcloud(clust9$words,clust9$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 9 Wordcloud", col.main = "grey14")
wordcloud(clust10$words,clust10$freq,max.words=300,colors=brewer.pal(8,"Dark2"),scale=c(3,0.5),random.order=F)
title("Cluster 10 Wordcloud", col.main = "grey14")
####################################
##Cluster 1 Rating Summary
####################################
ak<-data.frame(hotelreview,km$cluster)
# cluster1<-subset(ak,km$cluster==1)
# summary(cluster1)
cluster1<-subset(ak,ak$km.cluster==1)
summary(cluster1)
cl1corpous<-Corpus(VectorSource(cluster1$km.cluster==1))

inspect(cl1corpous)

rating5<-subset(cluster1,cluster1$Rating==5)
summary(rating5)
rating4<-subset(cluster1,cluster1$Rating==4)
summary(rating4)
rating3<-subset(cluster1,cluster1$Rating==3)
summary(rating3)
rating2<-subset(cluster1,cluster1$Rating==2)
summary(rating2)
rating1<-subset(cluster1,cluster1$Rating==1)
summary(rating1)




####################################
##Cluster 2 Rating Summary
####################################

cluster2<-subset(ak,ak$km.cluster==2)
summary(cluster2)
cl2corpous<-Corpus(VectorSource(cluster2$km.cluster==2))

inspect(cl2corpous)
rating5<-subset(cluster2,cluster2$Rating==5)
summary(rating5)
rating4<-subset(cluster2,cluster2$Rating==4)
summary(rating4)
rating3<-subset(cluster2,cluster2$Rating==3)
summary(rating3)
rating2<-subset(cluster2,cluster2$Rating==2)
summary(rating2)
rating1<-subset(cluster2,cluster2$Rating==1)
summary(rating1)

####################################
##Cluster 3 Rating Summary
####################################

cluster3<-subset(ak,ak$km.cluster==3)
summary(cluster3)
cl3corpous<-Corpus(VectorSource(cluster3$km.cluster==3))
inspect(cl3corpous)

rating5<-subset(cluster3,cluster3$Rating==5)
summary(rating5)
rating4<-subset(cluster3,cluster3$Rating==4)
summary(rating4)
rating3<-subset(cluster3,cluster3$Rating==3)
summary(rating3)
rating2<-subset(cluster3,cluster3$Rating==2)
summary(rating2)
rating1<-subset(cluster3,cluster3$Rating==1)
summary(rating1)


####################################
##Cluster 4 Rating Summary
####################################

cluster4<-subset(ak,ak$km.cluster==4)
summary(cluster4)
cl4corpous<-Corpus(VectorSource(cluster4$km.cluster==4))
inspect(cl4corpous)
rating5<-subset(cluster4,cluster4$Rating==5)
summary(rating5)
rating4<-subset(cluster4,cluster4$Rating==4)
summary(rating4)
rating3<-subset(cluster4,cluster4$Rating==3)
summary(rating3)
rating2<-subset(cluster4,cluster4$Rating==2)
summary(rating2)
rating1<-subset(cluster4,cluster4$Rating==1)
summary(rating1)


####################################
##Cluster 5 Rating Summary
####################################

cluster5<-subset(ak,ak$km.cluster==5)
summary(cluster5)
cl5corpous<-Corpus(VectorSource(cluster5$km.cluster==5))
inspect(cl5corpous)
rating5<-subset(cluster5,cluster5$Rating==5)
summary(rating5)
rating4<-subset(cluster5,cluster5$Rating==4)
summary(rating4)
rating3<-subset(cluster5,cluster5$Rating==3)
summary(rating3)
rating2<-subset(cluster5,cluster5$Rating==2)
summary(rating2)
rating1<-subset(cluster5,cluster5$Rating==1)
summary(rating1)



####################################
##Cluster 6 Rating Summary
####################################

cluster6<-subset(ak,ak$km.cluster==6)
summary(cluster6)
cl6corpous<-Corpus(VectorSource(cluster6$km.cluster==6))
inspect(cl6corpous)
rating5<-subset(cluster6,cluster6$Rating==5)
summary(rating5)
rating4<-subset(cluster6,cluster6$Rating==4)
summary(rating4)
rating3<-subset(cluster6,cluster6$Rating==3)
summary(rating3)
rating2<-subset(cluster6,cluster6$Rating==2)
summary(rating2)
rating1<-subset(cluster6,cluster6$Rating==1)
summary(rating1)



####################################
##Cluster 7 Rating Summary
####################################

cluster7<-subset(ak,ak$km.cluster==7)
summary(cluster7)
cl7corpous<-Corpus(VectorSource(cluster7$km.cluster==7))
inspect(cl7corpous)
rating5<-subset(cluster7,cluster7$Rating==5)
summary(rating5)
rating4<-subset(cluster7,cluster7$Rating==4)
summary(rating4)
rating3<-subset(cluster7,cluster7$Rating==3)
summary(rating3)
rating2<-subset(cluster7,cluster7$Rating==2)
summary(rating2)
rating1<-subset(cluster7,cluster7$Rating==1)
summary(rating1)

####################################
##Cluster 8 Rating Summary
####################################

cluster8<-subset(ak,ak$km.cluster==8)
summary(cluster8)
cl8corpous<-Corpus(VectorSource(cluster8$km.cluster==8))
inspect(cl8corpous)
rating5<-subset(cluster8,cluster8$Rating==5)
summary(rating5)
rating4<-subset(cluster8,cluster8$Rating==4)
summary(rating4)
rating3<-subset(cluster8,cluster8$Rating==3)
summary(rating3)
rating2<-subset(cluster8,cluster8$Rating==2)
summary(rating2)
rating1<-subset(cluster8,cluster8$Rating==1)
summary(rating1)
####################################
##Cluster 9 Rating Summary
####################################

cluster9<-subset(ak,ak$km.cluster==9)
summary(cluster9)
cl9corpous<-Corpus(VectorSource(cluster9$km.cluster==9))
inspect(cl9corpous)
rating5<-subset(cluster9,cluster9$Rating==5)
summary(rating5)
rating4<-subset(cluster9,cluster9$Rating==4)
summary(rating4)
rating3<-subset(cluster9,cluster9$Rating==3)
summary(rating3)
rating2<-subset(cluster9,cluster9$Rating==2)
summary(rating2)
rating1<-subset(cluster9,cluster9$Rating==1)
summary(rating1)



####################################
##Cluster 10 Rating Summary
####################################

cluster10<-subset(ak,ak$km.cluster==10)
summary(cluster10)
cl10corpous<-Corpus(VectorSource(cluster10$km.cluster==10))
inspect(cl10corpous)
rating5<-subset(cluster10,cluster10$Rating==5)
summary(rating5)
rating4<-subset(cluster10,cluster10$Rating==4)
summary(rating4)
rating3<-subset(cluster10,cluster10$Rating==3)
summary(rating3)
rating2<-subset(cluster10,cluster10$Rating==2)
summary(rating2)
rating1<-subset(cluster10,cluster10$Rating==1)
summary(rating1)



####################################
# LDA TOPIC MODELING : 
####################################
# function to get & plot the most informative terms by a specificed number
# of topics, using LDA
top_terms_by_topic_LDA <- function(reviewtext, # should be a columm from a dataframe
                                   plot = T, # return a plot? TRUE by defult
                                   nooftopics = 4) # number of topics (4 by default)
{    
  # create a corpus (type of object expected by tm) and document term matrix
  Corpus <- Corpus(VectorSource(reviewtext)) # make a corpus object
  DTM <- DocumentTermMatrix(Corpus) # get the count of words/document
  
  # remove any empty rows in our document term matrix 
  uniqueindexes <- unique(DTM$i) # get the index of each unique value
  DTM <- DTM[uniqueindexes,] # get a subset of only those indexes
  
  # preform LDA & get the words/topic in a tidy text format
  lda <- LDA(DTM, k = nooftopics,method = "Gibbs",  control = list(seed = 1234, burnin = 1000,
                                                                         thin = 100, iter = 1000))
  topics <- tidy(lda, matrix = "beta")
  
  # get the top ten terms for each topic
  topterms <- topics  %>% # take the topics data frame and..
    group_by(topic) %>% # treat each topic as a different group
    top_n(10, beta) %>% # get the top 10 most informative words
    ungroup() %>% # ungroup
    arrange(topic, -beta) # arrange words in descending informativeness
  
  if(plot == T){
    # plot the top ten terms for each topic in order
    topterms %>% # take the top terms
      mutate(term = reorder(term, beta)) %>% # sort terms by beta value 
      ggplot(aes(term, beta, fill = factor(topic))) + # plot beta by theme
      geom_col(show.legend = FALSE) + # as a bar plot
      facet_wrap(~ topic, scales = "free") + # which each topic in a seperate plot
      labs(x = NULL, y = NULL) + # no x label, change y label 
      coord_flip() # turn bars sideways
  }else{ 
  
    return(topterms)
  }
}
reviewsCorpus <- Corpus(VectorSource(hotelreview$Review)) 
reviewsDTM <- DocumentTermMatrix(reviewsCorpus)

# convert the document term matrix to a tidytext corpus
reviewsDTMtidy <- tidy(reviewsDTM)


customstopwords <- tibble(word = c("hotel", "room"))


# remove stopwords
reviewsDTMtidycleaned <- reviewsDTMtidy %>% # take our tidy dtm and...
  anti_join(stop_words, by = c("term" = "word")) %>% # remove English stopwords and...
  anti_join(customstopwords, by = c("term" = "word")) # remove my custom stopwords

# stem the words (e.g. convert each word to its stem, where applicable)
reviewsDTMtidycleaned <- reviewsDTMtidycleaned %>% 
  mutate(stem = wordStem(term))
# reconstruct our documents
cleaneddocuments <- reviewsDTMtidycleaned %>%
  group_by(document) %>% 
  mutate(terms = toString(rep(stem, count))) %>%
  select(document, terms) %>%
  unique()

# # reconstruct cleaned documents (so that each word shows up the correct number of times)
cleaneddocuments <- reviewsDTMtidycleaned %>%
  group_by(document) %>%
  mutate(terms = toString(rep(term, count))) %>%
  select(document, terms) %>%
  unique()

# check out what the cleaned documents look like (should just be a bunch of content words)
# in alphabetic order
head(cleaneddocuments)

top_terms_by_topic_LDA(cleaneddocuments$terms, nooftopics = 6)
