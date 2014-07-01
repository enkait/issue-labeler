echo "" >> results
echo "====================================================" >> results
cat run.sh >> results
echo "----------------------------------------------------" >> results
cat naivebayes.py >> results
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++" >> results
python naivebayes.py -train_file ../largedata/preprocessed/data_list -cv_file ../largedata/preprocessed/cv_list | tee -a results
