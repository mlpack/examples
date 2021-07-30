#!/usr/bin/env bash
# Spam Classification with mlpack on the command line

## Introduction

: <<'COMMENT'
In this tutorial, the mlpack command line interface will
be used to train a machine learning model to classify
SMS spam. It will be assumed that mlpack has been
successfully installed on your machine. The tutorial has
been tested in a linux environment. It is written in
bash - https://www.gnu.org/software/bash/

Download the dataset using the `download_data_set.py` python script in the 
tools directory.

Then run this example script using

  bash spam_classification.sh

If you are using a low power computer, you may want to run 
the script using

  nice -20 bash spam_classification.sh

so that it runs as a background task.
COMMENT

## Example

: <<'COMMENT'
As an example, we will train a machine learning model to classify 
spam SMS messages. We will use an example spam dataset in Indonesian 
provided by Yudi Wibisono.

We will try to classify a message as spam or ham by the number of 
occurences of a word in a message. We first change the file line 
endings, remove line 243 which is missing a label and then remove the 
header from the dataset. Then, we split our data into two files, labels 
and messages. Since the labels are at the end of the message, the message 
is reversed and then the label removed and placed in one file. The 
message is then removed and placed in another file.
COMMENT

tr '\r' '\n' < ../data/dataset_sms_spam_bhs_indonesia_v1/dataset_sms_spam_v1.csv > dataset.txt
sed '243d' dataset.txt > dataset1.csv
sed '1d' dataset1.csv > dataset.csv
rev dataset.csv | cut -c1  | rev > labels.txt
rev dataset.csv | cut -c2- | rev > messages.txt
rm dataset.csv
rm dataset1.csv
rm dataset.txt

: <<'COMMENT'
Machine learning works on numeric data, so we will use labels of 
1 for ham and 0 for spam. The dataset contains three labels, 0, 
normal sms (ham), 1, fraud (spam) and 2, promotion (spam). We will 
label all spam as 1, so promotions and fraud will be labelled as 1.
COMMENT

tr '2' '1' < labels.txt > labels.csv
rm labels.txt

: <<'COMMENT'
The next step is to convert all text in the messages to lower case 
and for simplicity remove punctuation and any symbols that are not 
spaces, line endings or in the range a-z (one would need expand 
this range of symbols for production use) 
COMMENT

tr '[:upper:]' '[:lower:]' < messages.txt > messagesLower.txt
tr -Cd 'abcdefghijklmnopqrstuvwxyz \n' < messagesLower.txt > messagesLetters.txt
rm messagesLower.txt

: <<'COMMENT'
We now obtain a sorted list of unique words used (this step may take 
a few minutes).
COMMENT

xargs -n1 < messagesLetters.txt > temp.txt
sort temp.txt > temp2.txt
uniq temp2.txt > words.txt
rm temp.txt
rm temp2.txt

: <<'COMMENT'
We then create a matrix, where for each message, the frequency of word 
occurrences is counted (more on this on Wikipedia, 
https://en.wikipedia.org/wiki/Tf–idf and 
https://en.wikipedia.org/wiki/Document-term_matrix ).
COMMENT


declare -a words=()
declare -a letterstartind=()
declare -a letterstart=()
letter=" "
i=0
lettercount=0
while IFS= read -r line; do
        labels[$((i))]=$line
        let "i++"
done < labels.csv
i=0
while IFS= read -r line; do
        words[$((i))]=$line
        firstletter="$( echo $line | head -c 1 )"
        if [ "$firstletter" != "$letter" ]
        then
                letterstartind[$((lettercount))]=$((i))
                letterstart[$((lettercount))]=$firstletter
                letter=$firstletter
                let "lettercount++"
        fi
        let "i++"
done < words.txt
letterstartind[$((lettercount))]=$((i))
echo "Created list of letters"

touch wordfrequency.txt
rm wordfrequency.txt
touch wordfrequency.txt
messagecount=0
messagenum=0
messages="$( wc -l messages.txt )"
i=0
while IFS= read -r line; do
        let "messagenum++"
        declare -a wordcount=()
        declare -a wordarray=()
        read -r -a wordarray <<< "$line"
        let "messagecount++"
        words=${#wordarray[@]}
        for word in "${wordarray[@]}"; do
                startletter="$( echo $word | head -c 1 )"
                j=-1
                while [ $((j)) -lt $((lettercount)) ]; do
                        let "j++"
                        if [ "$startletter" == "${letterstart[$((j))]}" ]
                        then
                                mystart=$((j))
                        fi
                done
                myend=$((mystart))+1
                j=${letterstartind[$((mystart))]}
                jend=${letterstartind[$((myend))]}
                while [ $((j)) -le $((jend)) ]; do
                        wordcount[$((j))]=0
                        if [ "$word" == "${words[$((j))]}" ]
                        then
                                wordcount[$((j))]="$( echo $line | grep -o $word | wc -l )"
                        fi
                        let "j++"
                done
        done
        for j in "${!wordcount[@]}"; do
                wordcount[$((j))]=$(echo " scale=4; $((${wordcount[$((j))]})) / $((words))" | bc)
        done
        wordcount[$((words))+1]=$((words))
        echo "${wordcount[*]}" >> wordfrequency.txt
        echo "Processed message ""$messagenum"
        let "i++"
done < messagesLetters.txt

# Create csv file
tr ' '  ',' < wordfrequency.txt > data.csv

: <<'COMMENT'
Since Bash is an interpreted language, this simple implementation can 
take up to 30 minutes to complete.
COMMENT

: <<'COMMENT'
Once the script has finished running, split the data into testing (30%) 
and training (70%) sets:
COMMENT

mlpack_preprocess_split                        \
    --input_file data.csv                      \
    --input_labels_file labels.csv             \
    --training_file train.data.csv             \
    --training_labels_file train.labels.csv    \
    --test_file test.data.csv                  \
    --test_labels_file test.labels.csv         \
    --test_ratio 0.3                           \
    --verbose

: <<'COMMENT'
Now train a Logistic regression model
(https://mlpack.org/doc/stable/cli_documentation.html#logistic_regression):
COMMENT

mlpack_logistic_regression --training_file train.data.csv    \
                           --labels_file train.labels.csv    \
                           --lambda 0.1                      \
                           --output_model_file lr_model.bin

: <<'COMMENT'
Finally we test our model by producing predictions,
COMMENT

mlpack_logistic_regression --input_model_file lr_model.bin    \
                           --test_file test.data.csv          \
                           --output_file lr_predictions.csv

: <<'COMMENT'
and comparing the predictions with the exact results,
COMMENT

export incorrect=$(diff -U 0 lr_predictions.csv test.labels.csv | grep '^@@' | wc -l)
export tests=$(wc -l < lr_predictions.csv)
echo "scale=2;  100 * ( 1 - $((incorrect)) / $((tests)))"  | bc

: <<'COMMENT'
This gives approximately 90% validation rate, similar to that 
obtained at 
https://towardsdatascience.com/spam-detection-with-logistic-regression-23e3709e522

The dataset is composed of approximately 50% spam messages, 
so the validation rates are quite good without doing much parameter tuning.
In typical cases, datasets are unbalanced with many more entries in 
some categories than in others. In these cases a good validation
rate can be obtained by mispredicting the class with a few entries. 
Thus to better evaluate these models, one can compare the number of 
misclassifications of spam, and the number of misclassifications of ham. 
Of particular importance in applications is the number of false 
positive spam results as these are typically not transmitted. The next 
portion of the script creates a confusion matrix.
COMMENT


declare -a labels
declare -a lr
i=0
while IFS= read -r line; do
        labels[i]=$line
        let "i++"
done < test.labels.csv
i=0
while IFS= read -r line; do
        lr[i]=$line
        let "i++"
done < lr_predictions.csv
TruePositiveLR=0
FalsePositiveLR=0
TrueZerpLR=0
FalseZeroLR=0
Positive=0
Zero=0
for i in "${!labels[@]}"; do
        if [ "${labels[$i]}" == "1" ]
        then
                let "Positive++"
                if [ "${lr[$i]}" == "1" ] 
                then
                        let "TruePositiveLR++"
                else
                        let "FalseZeroLR++"
                fi
        fi
        if [ "${labels[$i]}" == "0" ]
        then
                let "Zero++"
                if [ "${lr[$i]}" == "0" ]
                then
                        let "TrueZeroLR++"
                else
                        let "FalsePositiveLR++"
                fi
        fi

done
echo "Logistic Regression"
echo "Total spam"  $Positive
echo "Total ham"  $Zero
echo "Confusion matrix"
echo "             Predicted class"
echo "                 Ham | Spam "
echo "              ---------------"
echo " Actual| Ham  | " $TrueZeroLR "|" $FalseZeroLR
echo " class | Spam | " $FalsePositiveLR " |" $TruePositiveLR
echo ""

: <<'COMMENT'
You should get output similar to

 Logistic Regression
 Total spam 183
 Total ham 159
 Confusion matrix
                  Predicted  class
                -------------------
                 |  Ham     | Spam
 | Actual | Ham  |  128     | 26
 | class  | Spam |  31      | 157

which indicates a reasonable level of classification.
Other methods you can try in mlpack for this problem include:
* Naive Bayes
https://mlpack.org/doc/stable/cli_documentation.html#nbc
* Random forest
https://mlpack.org/doc/stable/cli_documentation.html#random_forest
* Decision tree
https://mlpack.org/doc/stable/cli_documentation.html#decision_tree
* AdaBoost
https://mlpack.org/doc/stable/cli_documentation.html#adaboost
* Perceptron
https://mlpack.org/doc/stable/cli_documentation.html#perceptron

To improve the error rating, you can try other pre-processing methods 
on the initial data set.  Neural networks can give up to 99.95% 
validation rates, see for example:

https://thesai.org/Downloads/Volume11No1/Paper_67-The_Impact_of_Deep_Learning_Techniques.pdf 
https://www.kaggle.com/kredy10/simple-lstm-for-text-classification
https://www.kaggle.com/xiu0714/sms-spam-detection-bert-acc-0-993

However, using these techniques with mlpack is best covered in another tutorial.

This tutorial is an adaptation of one that first appeared in the Fedora Magazine
https://fedoramagazine.org/spam-classification-with-ml-pack/
COMMENT
