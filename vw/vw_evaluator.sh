train=hugedata_vw
train_small=hugedata_vw_small
test_set=hugecv_vw
split=1

count=`wc -l $train | cut -f 1 -d " "`
split_size=$(((count + (split - 1)) / split))

function train {
    vw --cache_file train.cache --oaa 3 --loss_function logistic -f data.model --passes 10 --compressed -d $1 -p test.pred
    cat test.pred | cut -f 1 -d "." > got_test
    cat $1 | cut -f 1 -d " " > expected_test
    echo "After training for training data $1:" >> result
    python estimate_params.py >> result
}

function eval_test {
    vw -i data.model -t -p test.pred -d $1
    cat test.pred | cut -f 1 -d "." > got_test
    cat $1| cut -f 1 -d " " > expected_test
    echo "Testing on $1:" >> result
    python estimate_params.py >> result
}

echo "===============================================================" >> result
cat genstats.sh >> result

for i in `seq 1 $split`; do
    samples=$((split_size*i))
    rm train.cache
    rm test.pred
    rm data.model
    echo "+++++++++++++++++++++++++" >> result
    head -n $samples $train > $train_small
    train $train_small
    eval_test $train_small
    eval_test $test_set
done
