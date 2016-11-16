
#!/bin/bash
set -e
set -x

python Assignment5_SVM.py -q31 -fold 1 handwriting/train.data handwriting/train.labels -test handwriting/test.data handwriting/test.labels > Q3_1_a
python Assignment5_SVM.py -q32 -fold 5 madelon/madelon_train.data madelon/madelon_train.labels -test madelon/madelon_test.data madelon/madelon_test.labels > Q3_2
python Assignment_DTs.py -fold 1 handwriting/train.data handwriting/train.labels -test handwriting/test.data handwriting/test.labels > Q3_Ensemble_discrete
python Assignment_cont_DTs.py -fold 1 madelon/madelon_train.data madelon/madelon_train.labels -test madelon/madelon_test.data madelon/madelon_test.labels > Q3_Ensemble_continuous
