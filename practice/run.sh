#! /bin/bash
DESTPATH=`mktemp`
echo "" >> $DESTPATH
echo "====================================================" >> $DESTPATH
cat run.sh >> $DESTPATH
echo "----------------------------------------------------" >> $DESTPATH
cat naivebayes.py >> $DESTPATH
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $DESTPATH
python naivebayes.py -train_file ../largedata/lemmatized/data_list -cv_file ../largedata/lemmatized/cv_list | tee -a $DESTPATH
cat $DESTPATH >> results
