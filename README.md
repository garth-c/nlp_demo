# NLP Demo

In this NLP project, I am working in the world of natural language processing, combining sentiment analysis and cosine similarity to analyze text data. The project aims to extract valuable insights from textual content by determining sentiment polarity and measuring document similarity. Using NLP libraries in Python, I preprocess and clean the text data, preparing it for sentiment analysis. Through sentiment analysis, I will classify the sentiment of each document as positive, negative, or neutral, enabling us to understand the overall sentiment distribution in the dataset.

Next, I will implement cosine similarity, a widely used measure for text similarity. By converting the documents into numerical vectors using techniques like a Document Term Matrix (DTM), I will calculate the cosine similarity scores to identify documents that exhibit similar content.

With these analyses, the project offers valuable applications such as sentiment monitoring for customer reviews or social media sentiment tracking. Additionally, it aids in identifying similar documents for information retrieval and clustering tasks.

Specifically, the goal for this project is to demo NLP sentiment analysis capabilities in both Python and R through Python. The R portion will focus on the similarity scoring between the tweets and the Python section will focus on sentiment analysis. 

### Data set description

The data set used for this demo is the 2015 subset of tweets for airlines. The data set is from Kaggle.com and it is 14,519 records long and it contains publically available tweets regarding various aspects related to airlines. The tweets are all in English and the are in a single .csv file. A snippet of the file is shown below.

<img width="472" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/c005c2ea-ebf5-4432-ae4a-3e6ec91f4cef">

------------------------------------------------------------------------------------------------------

### Go back to my profile page
[garth-c profile page] (https://github.com/garth-c)

----------------------------------------------------------------------------------------------------------------------------------------

## roadmap for this demo
- set up the Pycharm environment
- read in the source data set
- lematize the tokens
- nltk sentiment model
- pytorch sentiment model
- apply torch model to the data
- textual similarity using R and Quanteda

------------------------------------------------------------------------------------------------------------------------

# set up the Pycharm environment

The code below is a snapshop of my session info


The code below is how I have set up my Pycharm environment

```
###~~~
#set up the computing environment
###~~~

import os
os.chdir('C:/Users/matri/Documents/my_documents/local_git_folder/python_code/nlp')
os.getcwd()

#data wrangling
import pandas as pd
print(pd.__version__)

#perform the sentiment work
import nltk
print(nltk.__version__)

#one time downloads if they are current
nltk.download() #download the nltk models
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

-------------------------------------------------------------------------------------------------------

# read in the source data

The next step is to read in the source data set and perform any needed data prep. The .csv file is read in and pushed to a Pandas data frame.

```
###~~~
#read in the source data set
###~~~

#source data file
docs = pd.read_csv('tweets.csv',
                   encoding='utf-8')
docs.info()
docs.describe()

#recast all characters as lower case
docs['text_lower'] = docs['text'].astype(str).str.lower()
```

A screen shot from the console of the import is below

<img width="171" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/6f3e00fb-ab18-48bc-874c-4c23d49fb9ed">

The only data prep step for the sentiment analysis that I performed was to lower case all of the text into a new column in the Pandas data frame. The output of this step is shown below. 

<img width="450" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/adf11310-f798-4e45-91da-c70ab9b219cd">

-----------------------------------------------------------------------------------

# lematize the tokens

The next step is to lematize the tokens. For review, a token is a single word in the text and the nltk library will do the tokenization automatically as part of the lematization process. This is all accomplished with a function in Python and the last part of this code uses the function to apply it to the lematized data and put the output into a new column.


```
###~~~
#lematize the tokens
###~~~

#import the word stemmer
from nltk.stem import WordNetLemmatizer

#instantiate the lemmatizer function
wordnet_lem = WordNetLemmatizer()

#define a function for the lemmatization
def lemmatize_words(text):
        words = text.split()
        words = [wordnet_lem.lemmatize(word,pos='v') for word in words]
        return ' '.join(words)

#apply the function to the data
docs['text_lem'] = docs['text_lower'].apply(lemmatize_words)
```

The output of this step is below.

<img width="656" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/aa4a0b0a-7b0d-438a-8aac-13fe1e5dfdfe">

----------------------------------------------------------------------------------

# nltk sentiment model

Now that the source data has been tokenized and lematized, the nltk sentiment function will determine the sentiment score. This is also accomplished with two Python functions: the first for applying the instantiated sentiment function and the second is a lambda function to apply the overall scoring method. The first function returns the polarity probabilities in a tuple with the relative probabilities for postive, neutral, and negative. Then the relative scores by label are put into their own column for futher analysis if needed. The last function determines the overall most likely sentiment score (label with the highest score) and puts that value into the 'sentiment' column. This is the column that is the final determination for this data set. 

```
###~~~
#perform the sentiment analysis
###~~~

#import the sentiment function
from nltk.sentiment import SentimentIntensityAnalyzer

#instantiate the sentiment function
analyzer = SentimentIntensityAnalyzer()

#perform the sentiment analysis
docs['polarity'] = docs['text_lem'].apply(lambda x: analyzer.polarity_scores(x))

#put scores into their own columns
docs[['neg','neu','pos','compound']] = pd.DataFrame(docs['polarity'].tolist(),
                                                    index=docs.index)
#set the overall sentiment score
docs['sentiment'] = docs['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')
```

The count of postive, neutral and negative sentiment scores from the nltk method is shown below:

```
#plotting of the sentiment counts
import matplotlib.pyplot as plt
counts = docs.sentiment.value_counts().plot(kind = 'bar')
for i in counts.containers:
    counts.bar_label(i, label_type = 'edge')
plt.show(block = True)
```

![nltk](https://github.com/garth-c/nlp_demo/assets/138831938/81baf8b1-4129-44fb-becd-3b1b8dfec6ff)


-----------------------------------------------------------------------------------------------------------------------------------------------

# pytorch sentiment model

The next step is to determine the sentiment analysis using Pytorch and a pre-trained model from NLP town. The computing enviroment is the same but additional libraries will need to be imported as shown below.

```
#transformers for language model
import transformers as transformers
print(transformers.__version__)

#import the transformers functions
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

Next the pre-trained model and tokenizer is downloaded and instantiated. A really good resource for these and many other NLP models is:
https://huggingface.co/models

This specific model returns a count of stars ranging from 1 to 5. The interpretation of the star counts is below.
1 = most negative
3 = most neutral
5 = most positive

```
###~~~
#instantiate the pre-trained models and calibrate
###~~~

#set up the tokenizer and download the latest pre-trained model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#set up the sentiment model and load the latest pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

Next, I test out the sentiment scoring model with a test sentense: "the flight was just alright" and I am expecting a neutral star count of 3 for this very neutral sentense. The model correctly labels this as a neutral statement:

<img width="460" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/fba5d21d-45ca-4f35-85de-2cb72ae3780c">

Swapping the test sentense out with "the flight was incredibly awesome!!" give the expected start rating of 5 which is the most positive rating.
<img width="452" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/cd0c7d8f-14b7-4b13-a0bb-2519142fe9b7">

Swapping the test sentense out with "the flight was the worst experience ever!!" give the expected start rating of 1 which is the most negative rating.
<img width="454" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/cd4ed828-b152-4bf7-9941-1bd7027f3419">

Now that the model seems to be calibrated well, it's time to apply the model to the source data.


----------------------------------------------------------------------------------------------------------------

# apply torch model to the data

Since Pytorch will only read the first 512 characters, the sentiment score will be based on this cut off value. The process is implemted using a function to apply the model to the data and put the output in star ratings into a new column in a Pandas data frame. This model is also applied to the lower cased data but no other data prep steps were applied. 

```
###~~~
#apply the working model to real data
###~~~

#create a function to calculate the sentiment and tag each document
def sentiment_score(doc_string):
    tokens = tokenizer.encode(doc_string, return_tensors = 'pt')
    results = model(tokens)
    return int(torch.argmax(results.logits))+1

#apply the function to the data frame
docs['sentiment_score'] = (docs['text_cleaned'].apply(lambda x: sentiment_score(x[:512])))
```

The output of the function is shown below in the Pandas data frame:

![torch](https://github.com/garth-c/nlp_demo/assets/138831938/2fa55bcf-f366-41ad-977f-cf7d0428351c)

Plot the counts by star ratings.

```
#plotting of the sentiment counts
import matplotlib.pyplot as plt
counts = docs.sentiment_score.value_counts().plot(kind = 'bar')
for i in counts.containers:
    counts.bar_label(i, label_type = 'edge')
plt.show(block = True)
```
The plot shows a much more negative sentiment than the nltk version. This also highlights that different sentiment models will have different conclusions on the same source data and that the most appropriate version for the project should be applied. 

![pytorch](https://github.com/garth-c/nlp_demo/assets/138831938/950fbb56-7176-459f-857b-99629aeb54c0)

--------------------------------------------------------------------


# textual similarity using R and Quanteda

Next to demo the ability to use R from within Python and Pycharm, I will process a document similarity function that will calculate the similarity score as a percent between the documents. This would be used to group like text together for further analysis. 


The R code that was used is shown below. The first step is to make sure that the R plug in is downloaded and installed in Pycharm. Then the plug in needs to be pointed to the location of the R executable file. After this is done, then the R computing environment needs to be set up from within Pycharm and the needed libraries installed.

```
###~~~
#set up the R computing environment
###~~~

#load the needed libs
library(tidyverse)
library(quanteda)
library(quanteda.textmodels)
library(quanteda.textplots)
library(quanteda.textstats)


#set the seed number
set.seed(12345)
```

Next, read in the source data and pre-process the file.

```
###~~~
#read in the source data set
###~~~

#read in the file .csv version
text_data <- utils::read.csv('C:/Users/matri/Documents/my_documents/local_git_folder/python_code/nlp/tweets.csv')

#validate the imported data set against the source data file
View(text_data)
names(text_data)
dim(text_data)
text_data$text[1] #validate the first record

#remove line breaks in the text
text_data_net <- as.data.frame(gsub(pattern = '\n',
                                    replacement = '',
                                    x = text_data$text))

#give a value column name
colnames(text_data_net) <- c('text')
text_data_net$text[1]

#preprocess all text to lower case
text_data_net$text_adj <- stringr::str_trim(stringr::str_squish(base::tolower(text_data_net$text)))
text_data_net$text_adj[1]

#bind with ID column if needed
text_data_net_all <- as.data.frame(cbind(text_data[,1], text_data_net[,2]))
colnames(text_data_net_all) <- c('doc_id','text')
text_data_net_all$text[2]

#look at the df object
View(text_data_net_all)
```

The next step is to create a corpus and start with the data prep portion. This consists of removing symbols, padding, etc. from the data set.

```
###~~~
#data prep
###~~~

#create a corpus
text_corp <- quanteda::corpus(x = text_data_net_all,
                              text_field = 'text')

#look at it
print(text_corp)
summary(text_corp, 25)

#tokenize the corpus
toks_text <- quanteda::tokens(x = text_corp,
                              what = 'word',
                              remove_symbols = TRUE,
                              remove_numbers = FALSE,
                              remove_url = TRUE,
                              remove_separators = TRUE,
                              split_hyphens = TRUE,
                              padding = FALSE,
                              verbose = TRUE,
                              remove_punct = TRUE) %>%
                       quanteda::tokens_tolower() %>%
                       quanteda::tokens_wordstem() %>%
                       quanteda::tokens_remove(pattern = stopwords('en'))

#look at it & validate
print(toks_text)
ntoken(toks_text)
ntype(toks_text)
toks_text[2] #look at an individual token for validation
```

Next a data frame matrix is created to hold the output. Also, additional data pre-processing happens by removing stop words well as any symbols or known characters that will bias the results.

```
#create a dfm
text_dfm <- quanteda::dfm(x = toks_text,
                          tolower = TRUE,
                          remove_padding = TRUE,
                          verbose = quanteda_options('verbose'))

#remove the stop words
text_dfm_net <-  quanteda::dfm_remove(x = text_dfm,
                                      stopwords('english'))

#additional removals if needed
text_dfm_net <- dfm_remove(text_dfm_net, c('©', 'ltd'))

#validate the dfm object
print(text_dfm_net)
ndoc(text_dfm_net)
nfeat(text_dfm_net)
topfeatures(text_dfm_net, 20)

```

Then the documument similarity score is calculated using the cosine similarity metric. 

```
###~~~
#document similarity data frame
###~~~

#document similarities as a percentage
dfm_similarities <- as.data.frame(quanteda.textstats::textstat_simil(x = text_dfm_net,
                                                                     y = NULL,
                                                                     margin = 'documents',
                                                                     method = 'cosine',
                                                                     min_simil = 0.95))

#look at the output
View(dfm_similarities)
```

Below is a snippet of the output from the document similarities function. The value in the column 'cosine' is the percent similar and the first two columns are the document numbers to reference.

<img width="163" alt="image" src="https://github.com/garth-c/nlp_demo/assets/138831938/1555b926-8154-4d70-ba88-7bbdca1442af">

For reference:
+ document 8034 = "wsj team offering flight access journal content http co ptsbka4cdj"
  
+ document 8167 = "wsj team offer flight access journal digital journal http co 2nzh3qoazo"
  
+ document 15 = "thanks"
  
+ document 3631 = "thanks"

So as can be seen above, document similarity is a powerful method to group like unstructured textual data. The business value of this is high as similar comments are able to be grouped and analyzed further in order to meet the project goals.

Thanks for reading this!

-----------------------------------------------------------------------------

### Go back to my profile page
[garth-c profile page] (https://github.com/garth-c)
