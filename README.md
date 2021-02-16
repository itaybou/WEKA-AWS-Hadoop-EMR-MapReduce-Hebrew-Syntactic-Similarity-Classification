# WEKA-AWS-Hadoop-EMR-MapReduce-Hebrew-Syntactic-Similarity-Classification
WEKA/AWS/Hadoop Elastic Map Reduce application to measure and classify word pair similarity 

Project GitHub repository link:  
https://github.com/itaybou/WEKA-AWS-Hadoop-EMR-MapReduce-Syntactic-Similarity-Classification

- Google English Syntactic Biarcs path: **https://storage.googleapis.com/books/syntactic-ngrams/index.html**
- Assignment Syntactic Biarcs S3 path: **s3://assignment3dsp/biarcs/**

## Created by:  
### Itay Bouganim : 305278384  
### Sahar Vaya : 205583453  
## 
## Table of contents
* [General info and stages](#general-info-and-stages)
* [Setup](#setup)
* [Instructions](#Instructions)
* [Project workflow and Map-Reduce design](#project-workflow-and-Map-Reduce-design)
* [Communication And Statistics](#communication-and-statistics)
* [Classification Results](#classification-results)
* [Results Analysis](#results-analysis)


## General Info And Stages

This assignment follows the paper from Ljubešić et al.: [Comparing Measures of Semantic Similarity](https://github.com/itaybou/WEKA-AWS-Hadoop-EMR-MapReduce-Syntactic-Similarity-Classification/blob/main/resources/comparing-measures-of-semantic-similarity.pdf) with some modifications.  
In this assignment we will perform **3** stages in order to determine whether two words are similar:
1. We will generate 4 syntactic relatedness co-occurrence vectors for every lexeme from the given Google English Syntactic Biarcs corpus.<br/>The features for each lexeme will be pairs of the feature word and the sentence dependency (e.g subject, noun etc.).<br/>Each vector will contain different ***meausre of assocation*** between the lexeme and the syntactic features related to that lexeme in the syntactic sentence tree so that each vector will be a syntactic representation of the lexeme by using the features it is connected to in the corpus.
2. We will then use a pre-defined golden standard word-pairs set and ***measure the similarity*** of their representing 4 co-occurrence vectors calculated in the previous stage.<br/>The similarity will be measured by using 6 different methods used to calculate the distance between those vectors and the output will be a vector of size 24 (4 association methods x 6 similarity methods) representing the similarity between the word pair.
3. Then we will use a classical ML Random Forest algorithm (Supervised learning) by using WEKA Java package in order to ***classify*** the similarity vectors produced and compare the results to the pre-classified word-pairs in the golden standard.

We will be using Amazon Elastic Map-Reduce (EMR) in order to compute the first two stages out of the input corpus.  
The produced output of the Map-Reduce stages will be the word-pair similaritry vectors.  
For the third stage we will use the Java WEKA package in order to train and evaluate our classifier based on the labeld word-pairs given in the golden standard.


#### Stage 1 - Measures Of Association With Context
Measures of association with context are used to compute values that are included in the co-occurrence vectors.  
These values are based on the frequencies of lexemes and features extracted from corpus.
In this assignment we will use 4 different methods to calculate the measures of association producing 4 different co-occurrence vectors for each lexeme in the corpus with values relevant to each of the association methods.

The 4 association methods used are:
1. **Plain Frequency** - ```count(L=l, F=f)``` the amount of times that lexeme ```l``` appeared with feature ```f``` in the corpus.
2. **Relative Frequency** - ```P(F=f | L=l)``` the amount of times that lexeme ```l``` appeared with feature ```f``` divided by the total amount of appearences of lexeme ```l``` (So we will get a normalized vector relative to the appearences of the lexeme).<br/>Meaning that ```P(F=f | L=l) = count(L=l, F=f) / count(L=l)``` and ```count(L=l)``` is the amount of time that lexeme ```l``` appeared in the corpus.
3. **Pointwise Mutual Information (PMI)** - ```log_2(P(L=l, F=f) / (P(L=l) * P(F=f)))``` where ```P(L=l, F=f) = count(L=l, F=f) / count(L)```,  ```count(L)``` is the total number of appearences of different lexemes in the corpus, ```P(L=l)``` is the frequency of lexeme ```l``` in the corpus (```count(L=l) / count(L)```) and ```P(F=f)``` is the frequency of feature ```f``` in the corpus (```count(F=f) / count(F)```).<br/>This measure computes how often a lexeme ```l``` and a feature ```f``` co-occur, compared with what would be expected if they were independent.
4. **T-Test Measure** - ```(P(L=l, F=f) - (P(L=l) * P(F=f))) / sqrt((P(L=l) * P(F=f)))``` where ```P(L=l, F=f) = count(L=l, F=f) / count(L)```, ```P(L=l)``` is the frequency of lexeme ```l``` in the corpus (```count(L=l) / count(L)```) and ```P(F=f)``` is the frequency of feature ```f``` in the corpus (```count(F=f) / count(F)```).<br/>Is a statistic measurment which attempts to capture the same intuition as pointwise mutual information statistic which computes the difference between observed and expected means, normalized by the variance.

#### Stage 2 - Measures Of Vector Similarity
Measures of vector similarity are used to compare two vectors, ie. lexemes built from association measures for features used to describe the lexemes.
In this assignment we will use 6 different methods to calculate the similarity between co-occurrence vectors producing, for each lexeme in the golden standard, a single vector with length 24 (4 measure association x 6 similarity measures).

The 6 similarity measured used are:  
***```l1``` and ```l2``` are the co-occurrence vectors corrosponding with word 1 and word 2 in the compared word pair.***
1. **Manhattan Distance** - Calculates the distance between vectors on all dimensions: ```sum(abs(l1[i] - l2[i]))```.
2. **Euclidean Distance** - Measures the geometric distance between the two vectors: ```sqrt(sum((l1[i] - l2[i])^2))```.
3. **Cosine Distance** - A measure used in information retrieval is the dot product operator from linear algebra.<br/>If the vectors are normalized, that measure is equal to the cosine between the two vectors.<br/>The cosine similarity measure is computed by ```(sum(l1[i] * l2[i]) / (sqrt(sum(l1[i]^2)) * sqrt(sum(l2[i]^2))))```.
4. **Jaccard Measure** - A measure that is derived from information retrieval.<br/>Divides the number of equal features with the number of features in general and extended to vectors with weighted associations as follows: ```(sum(min(l1[i], l2[i])) / sum(max(l1[i], l2[i])))```.
5. **Dice Measure** - Very similar to the Jaccard measure and is also introduced from information retrieval.<br/>The equivalent of the Dice measure for weighted associations is computed as follows: ```((2 * sum(min(l1[i], l2[i]))) / sum(l1[i] + l2[i]))```.
6. **Jensen-Shannon Divergence** - From the family of information-theoretic distributional similarity measures.<br/>The intuition of these methods is that two vectors ```l1``` and ```l2``` are similar to the extent that their probability distributions ```P(F=f|l1)``` and ```P(F=f|l2)``` are similar.<br/>We compare the two probability distributions by using the **Kullback-Leibler divergence** that is computed by: ```D(l||j) = sum(l[i] * log_10(l[i] / j[i]))```.<br/>To bypass the negative property (Since our co-occurrence vectors are inheritably sparse, meaning they contain a lot of zeros) we use the **Jensen-Shannon** divergence which is computed as follows: ```D(l1||(l1 + l2) / 2) + D(l2||(l1 + l2) / 2)```.<br/>by using both formulas we get that the Jensen-Shannon Divergence of vectors ```l1``` and ```l2``` is equal to: ```sum(l1[i] * log(l1[i] / ((l1[i] + l2[i]) / 2))) + sum(l2[i] * log(l2[i] / ((l1[i] + l2[i]) / 2)))```

The output vector for each lexeme will have the following shape:
```
1 plain frequency vector - Manhattan distance
2 plain frequency vector - Euclidean distance
3 plain frequency vector - Cosine distance
4 plain frequency vector - Jaccard measure
5 plain frequency vector - Dice measure
6 plain frequency vector - Jensen-Shannon divergence
7 relative frequency vector - Manhattan distance
8 relative frequency vector - Euclidean distance
9 relative frequency vector - Cosine distance
10 relative frequency vector - Jaccard measure
11 relative frequency vector - Dice measure
12 relative frequency vector - Jensen-Shannon divergence
13 pointwise mutual information vector - Manhattan distance
14 pointwise mutual information vector - Euclidean distance
15 pointwise mutual information vector - Cosine distance
16 pointwise mutual information vector - Jaccard measure
17 pointwise mutual information vector - Dice measure
18 pointwise mutual information vector - Jensen-Shannon divergence
19 t-test statistic vector - Manhattan distance
20 t-test statistic vector - Euclidean distance
21 t-test statistic vector - Cosine distance
22 t-test statistic vector - Jaccard measure
23 t-test statistic vector - Dice measure
24 t-test statistic vector- Jensen-Shannon divergence
```

#### Stage 3 - Similarity Vectors Classification (Similar or Not-Similar)
In this stage we use Random Forest classifier in order te perform a supervised learning task on our pre-labeled word pairs from the golden standard.  
The input for our RF classifier is the similarity vectors for the word pairs from the golden standard produced in the previous stage and the output is the **True-Positive, False-Positive, True-Negative and True-Positive rates**.
We use **10-fold cross validation** method in order to evaluate our model performance (which trains 10 iterations of the model by using a different portion of the training data as testing data in each iteration).
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)
We also output the **Precision, Recall and F1 measurments** where:
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)

### Additonal information
EC2 instances used:
Workers - 
 * Machine types - (64-bit x86) type: M4_LARGE
 
## Setup
1. Install aws cli in your operating system, for more information click here :
https://aws.amazon.com/cli/

2. Configure your amazon aws credentials in your .aws directory, alternatively you can set your credentials by using aws cli : 
write in your cmd - "aws config".


## Instructions

1. Inside the project directory compile the project using the command : ```mvn package```.

2. Create in the project target directory file named "inputs.txt".

3. Create input bucket and output bucket (you can use the same bucket and create only one bucket) in AWS S3.

4. 

5. Fill 
```
<input-bucket> <input-jar-file-name> <input-golden-standard> <upload-jar-and-golden-standard>
<corpus-input-path>
<output-bucket>
<corpus-files-count>
<worker-instance-count>
<calculate-measures> <measures-path>
<output-co-occurrence-vectors>
<run-classifier> <classifier-output-path> <optional-classifier-input-path>
<delete-after-finished>
```
- ```<input-bucket>``` - Is the bucket the jar file ```<input-jar-file-name>``` is located in.  
- ```<output-bucket>``` - Is the bucket the job will store outputs in **(Will be deleted after the job is completed!)**, Can be the same as input bucket.  
- ```<worker-instance-count (0 < x < 10)>``` - The EC2 instance count that will be used for the map-reduce job (value between 0 excluding and 9 including)  
- ```<use-local-aggregation (true or false)>``` - Wether the job will use Combiners for local aggregation in order to lower network overhead.  
- ```<single-file (true or false)>``` - Whether to output single sorted output file (slower) or multiple sorted output files (faster).  

5. Make sure your input text file located in the project target directory or in the same directory as the WordPedictionRunner jar file.

6. The application should be run as follows:  
	```java -jar WordPedictionRunner.jar ```  

***IMPORTANT NOTES:***
 - The application automatically uploads the input jar provided in the ```inputs.txt``` file to the input bucket provided in the ```inputs.txt``` file.
 - When the job is finished the output result and log-files will be automatically downloaded to the directory the ```java -jar WordPedictionRunner.jar ``` was ran from.
 - The output bucket provided in the ```inputs.txt``` file will be automatically deleted.

## Communication And Statistics:

Using the python script in the statistics directory the following statstics charts were produced from the output log-files:

### Using 2 Files from the Google English Syntactic Biarcs corpus
Total lexemes read from corpus: 704071870 (```count(L)```)  
Total features read from corpus: 802056549 (```count(F)```)

Total number of word-pairs in the golden standard: 13254

#### Input Output Records Statistics:
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)

|    | Status   | Statistic              | Stage                                          |    Value |
|----|----------|------------------------|------------------------------------------------|----------|
|  0 | 2files   | Map input records      | Parse Syntactic Dependencies                   | 33382028 |
|  1 | 2files   | Map output records     | Parse Syntactic Dependencies                   | 73804771 |
|  2 | 2files   | Combine input records  | Parse Syntactic Dependencies                   | 73804771 |
|  3 | 2files   | Combine output records | Parse Syntactic Dependencies                   |  7594149 |
|  4 | 2files   | Reduce input records   | Parse Syntactic Dependencies                   |  7594149 |
|  5 | 2files   | Reduce output records  | Parse Syntactic Dependencies                   |  4237086 |
|  6 | 2files   | Map input records      | Order And Count Lexeme Feature                 |  4237086 |
|  7 | 2files   | Map output records     | Order And Count Lexeme Feature                 |  4237086 |
|  8 | 2files   | Combine input records  | Order And Count Lexeme Feature                 |        0 |
|  9 | 2files   | Combine output records | Order And Count Lexeme Feature                 |        0 |
| 10 | 2files   | Reduce input records   | Order And Count Lexeme Feature                 |  4237086 |
| 11 | 2files   | Reduce output records  | Order And Count Lexeme Feature                 |  4140401 |
| 12 | 2files   | Map input records      | Calculate Measures Of Association With Context |  4140401 |
| 13 | 2files   | Map output andrecords     | Calculate Measures Of Association With Context |  4140401 |
| 14 | 2files   | Combine input records  | Calculate Measures Of Association With Context |        0 |
| 15 | 2files   | Combine output records | Calculate Measures Of Association With Context |        0 |
| 16 | 2files   | Reduce input records   | Calculate Measures Of Association With Context |  4140401 |
| 17 | 2files   | Reduce output records  | Calculate Measures Of Association With Context |  3732246 |
| 18 | 2files   | Map input records      | Calculate Measures Of Vector Similarity        |  3732246 |
| 19 | 2files   | Map output records     | Calculate Measures Of Vector Similarity        |  7604638 |
| 20 | 2files   | Combine input records  | Calculate Measures Of Vector Similarity        |        0 |
| 21 | 2files   | Combine output records | Calculate Measures Of Vector Similarity        |        0 |
| 22 | 2files   | Reduce input records   | Calculate Measures Of Vector Similarity        |  7604638 |
| 23 | 2files   | Reduce output records  | Calculate Measures Of Vector Similarity        |    13245 |

#### Bytes Records Statistics:
![No Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_No_Combiner.png)

|    | Status   | Statistic            | Stage                                          |      Value |
|----|----------|----------------------|------------------------------------------------|------------|
|  0 | 2files   | Map output bytes     | Parse Syntactic Dependencies                   | 1558130249 |
|  1 | 2files   | Reduce shuffle bytes | Parse Syntactic Dependencies                   |   91646025 |
|  2 | 2files   | Bytes Read           | Parse Syntactic Dependencies                   |          0 |
|  3 | 2files   | Bytes Written        | Parse Syntactic Dependencies                   |   33253030 |
|  4 | 2files   | Map output bytes     | Order And Count Lexeme Feature                 |  193711966 |
|  5 | 2files   | Reduce shuffle bytes | Order And Count Lexeme Feature                 |   62976319 |
|  6 | 2files   | Bytes Read           | Order And Count Lexeme Feature                 |   33253030 |
|  7 | 2files   | Bytes Written        | Order And Count Lexeme Feature                 |   34643634 |
|  8 | 2files   | Map output bytes     | Calculate Measures Of Association With Context |  215170779 |
|  9 | 2files   | Reduce shuffle bytes | Calculate Measures Of Association With Context |   74235914 |
| 10 | 2files   | Bytes Read           | Calculate Measures Of Association With Context |   34643634 |
| 11 | 2files   | Bytes Written        | Calculate Measures Of Association With Context |  133121186 |
| 12 | 2files   | Map output bytes     | Calculate Measures Of Vector Similarity        |  636086316 |
| 13 | 2files   | Reduce shuffle bytes | Calculate Measures Of Vector Similarity        |  255931083 |
| 14 | 2files   | Bytes Read           | Calculate Measures Of Vector Similarity        |  133121186 |
| 15 | 2files   | Bytes Written        | Calculate Measures Of Vector Similarity        |    5762026 |

### Using 14 Files from the Google English Syntactic Biarcs corpus
Total lexemes read from corpus: 8004073453 (```count(L)```)  
Total features read from corpus: 9420023809 (```count(F)```)

Total number of word-pairs in the golden standard: 13254

#### Input Output Records Statistics:
![Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_Combiner.png)

|    | Status   | Statistic              | Stage                                          |     Value |
|----|----------|------------------------|------------------------------------------------|-----------|
|  0 | 14files  | Map input records      | Parse Syntactic Dependencies                   | 247503741 |
|  1 | 14files  | Map output records     | Parse Syntactic Dependencies                   | 732517441 |
|  2 | 14files  | Combine input records  | Parse Syntactic Dependencies                   | 732517441 |
|  3 | 14files  | Combine output records | Parse Syntactic Dependencies                   |  72358125 |
|  4 | 14files  | Reduce input records   | Parse Syntactic Dependencies                   |  72358125 |
|  5 | 14files  | Reduce output records  | Parse Syntactic Dependencies                   |  23517782 |
|  6 | 14files  | Map input records      | Order And Count Lexeme Feature                 |  23517782 |
|  7 | 14files  | Map output records     | Order And Count Lexeme Feature                 |  23517782 |
|  8 | 14files  | Combine input records  | Order And Count Lexeme Feature                 |         0 |
|  9 | 14files  | Combine output records | Order And Count Lexeme Feature                 |         0 |
| 10 | 14files  | Reduce input records   | Order And Count Lexeme Feature                 |  23517782 |
| 11 | 14files  | Reduce output records  | Order And Count Lexeme Feature                 |  23169981 |
| 12 | 14files  | Map input records      | Calculate Measures Of Association With Context |  23169981 |
| 13 | 14files  | Map output records     | Calculate Measures Of Association With Context |  23169981 |
| 14 | 14files  | Combine input records  | Calculate Measures Of Association With Context |         0 |
| 15 | 14files  | Combine output records | Calculate Measures Of Association With Context |         0 |
| 16 | 14files  | Reduce input records   | Calculate Measures Of Association With Context |  23169981 |
| 17 | 14files  | Reduce output records  | Calculate Measures Of Association With Context |  21641792 |
| 18 | 14files  | Map input records      | Calculate Measures Of Vector Similarity        |  21641792 |
| 19 | 14files  | Map output records     | Calculate Measures Of Vector Similarity        |  49972240 |
| 20 | 14files  | Combine input records  | Calculate Measures Of Vector Similarity        |         0 |
| 21 | 14files  | Combine output records | Calculate Measures Of Vector Similarity        |         0 |
| 22 | 14files  | Reduce input records   | Calculate Measures Of Vector Similarity        |  49972240 |
| 23 | 14files  | Reduce output records  | Calculate Measures Of Vector Similarity        |     13254 |

#### Bytes Records Statistics:
![No Combiner Input Output Records](https://github.com/itaybou/AWS-Hadoop-EMR-MapReduce-Hebrew-3gram-deleted-estimation/blob/main/statistics/In_Out_No_Combiner.png)

|    | Status   | Statistic            | Stage                                          |       Value |
|----|----------|----------------------|------------------------------------------------|-------------|
|  0 | 14files  | Map output bytes     | Parse Syntactic Dependencies                   | 15301600686 |
|  1 | 14files  | Reduce shuffle bytes | Parse Syntactic Dependencies                   |   856642242 |
|  2 | 14files  | Bytes Read           | Parse Syntactic Dependencies                   |           0 |
|  3 | 14files  | Bytes Written        | Parse Syntactic Dependencies                   |   183754451 |
|  4 | 14files  | Map output bytes     | Order And Count Lexeme Feature                 |  1097072386 |
|  5 | 14files  | Reduce shuffle bytes | Order And Count Lexeme Feature                 |   336855026 |
|  6 | 14files  | Bytes Read           | Order And Count Lexeme Feature                 |   183754451 |
|  7 | 14files  | Bytes Written        | Order And Count Lexeme Feature                 |   193350284 |
|  8 | 14files  | Map output bytes     | Calculate Measures Of Association With Context |  1241206109 |
|  9 | 14files  | Reduce shuffle bytes | Calculate Measures Of Association With Context |   422812654 |
| 10 | 14files  | Bytes Read           | Calculate Measures Of Association With Context |   193350284 |
| 11 | 14files  | Bytes Written        | Calculate Measures Of Association With Context |   782690392 |
| 12 | 14files  | Map output bytes     | Calculate Measures Of Vector Similarity        |  4136504326 |
| 13 | 14files  | Reduce shuffle bytes | Calculate Measures Of Vector Similarity        |  1742580460 |
| 14 | 14files  | Bytes Read           | Calculate Measures Of Vector Similarity        |   782690392 |
| 15 | 14files  | Bytes Written        | Calculate Measures Of Vector Similarity        |     6584955 |


## Word Analysis

| 2gram        | על קבר                                                                                                     | אהב את                                                                                                                                                                                                                                                   | אולי לא                                    | כבר במאה                                                                          | מה שהאדם            |
|--------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|-----------------------------------------------------------------------------------|---------------------|
| 1 Prediction | על קבר רחל                                                                                                 | אהב את יוסף                                                                                                                                                                                                                                              | אולי לא היה                                | כבר במאה השלישית                                                                  | מה שהאדם עושה       |
| 2 Prediction | על קבר אביו                                                                                                | אהב את המלאכה                                                                                                                                                                                                                                            | אולי לא היתה                               | כבר במאה העשירית                                                                  | מה שהאדם הוא        |
| 3 Prediction | על קבר שמואל                                                                                               | אהב את זה                                                                                                                                                                                                                                                | אולי לא היו                                | כבר במאה התשע                                                                     | מה שהאדם צריך       |
| 4 Prediction | על קבר הצדיק                                                                                               | אהב את עשו                                                                                                                                                                                                                                               | אולי לא פחות                               | כבר במאה הרביעית                                                                  | מה שהאדם יכול       |
| 5 Prediction | על קבר האחים                                                                                               | אהב את אשתו                                                                                                                                                                                                                                              | אולי לא הייתי                              | כבר במאה התשיעית                                                                  | מה שהאדם משיג       |
|              |                                                                                                            |                                                                                                                                                                                                                                                          |                                            |                                                                                   |                     |
| Decision     | We can see that the decision that was made here is correct since the first prediction is a common sentence | We can see that the prediction made here is not as we would expect. we would expect the most common prediction for someone to love his wife and not joseph. The possible reason is that old phrases are weighted the same as newer more updated phrases. | Here the prediction is as we would expect. | Here we would expect a more recent century to appear in in the top 5 predictions. | As we would expect. |

| 2gram        | בדיוק באותה                                    | תשובות על                                                                      | תלויה על                                              | על חלק                                                                     | ולא תוסיף                                                                                                                                           |
|--------------|------------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 Prediction | בדיוק באותה מידה                               | תשובות על שאלות                                                                | תלויה על בלימה                                        | על חלק מן                                                                  | ולא תוסיף קום                                                                                                                                       |
| 2 Prediction | בדיוק באותה צורה                               | תשובות על השאלות                                                               | תלויה על הקיר                                         | על חלק גדול                                                                | ולא תוסיף לנפול                                                                                                                                     |
| 3 Prediction | בדיוק באותה שעה                                | תשובות על כל                                                                   | תלויה על קיר                                          | על חלק ניכר                                                                | ולא תוסיף עוד                                                                                                                                       |
| 4 Prediction | בדיוק באותה דרך                                | תשובות על בני                                                                  | תלויה על חוט                                          | על חלק זה                                                                  | ולא תוסיף לדאבה                                                                                                                                     |
| 5 Prediction | בדיוק באותה תקופה                              | תשובות על שאלותיו                                                              | תלויה על צווארו                                       | על חלק של                                                                  | ולא תוסיף עצב                                                                                                                                       |
|              |                                                |                                                                                |                                                       |                                                                            |                                                                                                                                                     |
| Decision     | A very good prediction was given in this case. | As we expect all  top 5 prediction are about answers to some form of questions | The first prediction is a very common  hebrew phrase. | Not as we expected we would expect the 4th prediction to be the first one. | We would expect the 3rd prediction to be the first and the 4th prediction to be switched with the fifth one since the fifth is more common nowdays. |


## Project workflow and Map-Reduce design

We have only one step that include 5 map-reduce jobs, the following jobs are :

1. **Split corpus** - 
	This map-reduce job firstly filters all the three grams in the corpus by the following regular expression:
	```
	^(?:[א-ת]+[א-ת\\d+ ]+|[א-ת\\d+ ]+[א-ת]+|[א-ת\\d+ ]+[א-ת]+[א-ת\\d+ ]+)$"
	```
	with this filter we will map only three grams that start with hebrew characters and numbers.
	The map function will split the corpus logically and not physically by the even and odd line id's, for example, if the input of the map is:
	```
	1.How are you	1975	400 (400 is the occurences of the three gram in 1975)
	2.How are you	2020	300 (300 is the occurences of the three gram in 2020)
	3.How are you	1985	100 (100 is the occurences of the three gram in 1985)
	...
	```
	The output of the map function will be :
	```
	How are you	1	400,100 (1 indicating that this three gram is in the second part of the corpus)
	How are you	0	300 (1 indicating that this three gram is in the first part of the corpus)
	...
	```

	After mapping each three gram to the occurences in each part of the corpus, the reduce function will sum all the occurences in the first and the second 	part and the reducer output will be :
	```
	How are you	300	500 (300 is the occurences in the first part of the corpus and 500 is the occurences in the second part)
	...
	```

2. **Aggregate Nr Tr** - 
	This map-reduce job aggregates the sums of 4 values in the deleted estimation formula: Nr0, Nr1, Tr01, Tr10 by using
	the input of the splitted corpus:
	```
	How are you	300	500 (300 is the occurences in the first part of the corpus and 500 is the occurences in the second part)
	How are them	300	400 (300 is the occurences in the first part of the corpus and 400 is the occurences in the second part)
	```
	The output of the mapping for the Nr calculation will be:
	```
	Nr0	300	1,1
	Nr1	500	1
	Nr1	400	1
	```
	The output of the mapping for the Tr calculation will be:
	```
	Tr01	300	500,400
	Tr10	500	300
	Tr10	400	300
	```
	After mapping each occurences value to corrosponding Nr/Tr values the reducer will sum the values and the output will be:
	- For Nr:
	```
	Nr0	300	2
	Nr1	500	1
	Nr1	400	1
	```
	- For Tr:
	```
	Tr01	300	900
	Tr10	500	300
	Tr10	400	300
	```

3. **Join Nr Tr with 3grams** -
	The goal of this job is to join the 3grams with their corrosponding Nr and Tr values.
	The purpose of the job is to avoid storing a list of 3grams for each occurences value in the previous job, by doing that we remove all local memory usage 	  that is input dependant.
	This job has two mappers and one reducer.
	The first mapping function recieves the input from the split corpus job and outputs 4 values for each value as follows:
	```
	300	THREE_GRAM	How are you	Nr0 (300 is the first split occurences, THREE_GRAM enum indicator and Nr0 indicator for the first split)
	500	THREE_GRAM	How are you	Nr1
	300	THREE_GRAM	How are you	Tr01
	500	THREE_GRAM	How are you	Tr10
	...
	```
	The second mapping function recieves the input from the aggregate Nr Tr job and maps twice to the following outputs:
	- For Nr:
	```
	300	AGGREGATED	2	Nr0 (300 is the first split occurences, AGGREGATED enum indicator and Nr0 indicator for the first split)
	500	AGGREGATED	1	Nr1
	400	AGGREGATED	1	Nr1
	```
	- For Tr:
	```
	300	AGGREGATED	900	Tr01 (300 is the first split occurences, AGGREGATED enum indicator and Tr01 indicator for the first split)
	500	AGGREGATED	300	Tr10
	400	AGGREGATED	300	Tr10
	```
	After mapping each occurences value the reducer will join the occurences with the corrosponding 3grams.
	The input for the reduces will be sorted by the by the enum indicator and the occurences value and and grouped by the occurences value.
	The output of the reducer will be:
	```
	How are you	Nr0	2
	How are you	Nr1	1
	How are them	Nr1	1
	How are them	Nr0	2
	How are you	Tr01	900
	How are you	Tr10	300
	How are them	Tr01	900
	How are them	Tr10	400
	```
4. **Calculate deleted estimation: (Tr01 + Tr10) / (N * (Nr0 + Nr1))** -
	The goal of this job is to calculate the deleted estimation value for each 3gram.
	The mapping function return the 3grams with the operation and aggregated value (same as previous job).
	The reducer function recieves the total Ngram value N, captures the Nr0, Nr1, Tr01, Tr10 from the mapping function and emits the calculation of the deleted estimation formula. The output will look like:
	```
	How are you	<probability> (Where probability is in range [0, 1])
	How are them	<probability>
	```
	
5. **Sort deleted estimation output** -
	This goal of this job is to emit the deleted estimation values calculated in the previous in a sorted fashion so that it is sorted by:
	1. The first word of the 3gram.
	2. If first words are equal sort by second word of the three gram.
	3. If first 2 words from the 3 gram are equal sort by the probability value.
	(The output will be sorted by first two words alphabetically ascending and by the probabilities descending)
	
	If true flag for single file output is given in the ```inputs.txt``` file (expanded on later in this readme) than the job will use one reducer to output one sorted file. Otherwise, the output will be multiple sorted files.
	

### Local aggregation using Combiners
If true flag for local aggregation is given in the ```inputs.txt``` file (expanded on later in this readme) than the job will use combiner in order to optimize the redducer job where possible.  
The following jobs include an optional Combiner:  
**Split corpus** and **Aggregate Nr Tr**.  
Both use the combiner to locally aggregate values (Corpus split aggregation and Nr/Tr values aggragation) before passing them to the reducer.  
Statistics for the combiner usage difference can be found in the [Statistics](#Statistics) section.  

The following jobs do not include optional Combiner:
- **Join Nr Tr with 3grams -** Join operation only, No use for combiner.
- **Calculate deleted estimation -** Did not use Combiner in this map reduce job since we perform
	division operation in order to calculate the deleted estimation probability
	which is not an associative operation.
- **Sort deleted estimation output -** Sort operation only, no use for combiner.

## Examples And Resources
- After compiling the project - project JAR files can be found in the projects target directory.
- Example for the ```inputs.txt``` text file needed to run the project can be found in the directory of the project.

