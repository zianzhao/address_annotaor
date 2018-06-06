# address_annotaor
High recall and precision information extraction for Chinese postal address via hybrid model and active learning.

<h2>Requirements:</h2>

python (2.7)

gensim (2.3.0)

numpy (1.13.1)

PyMySQL (0.7.9)

python-Levenshtein (0.12.0)

scikit-learn (0.18.2)

sklearn-crfsuite (0.3.6)      

<h2>Training:</h2>

- Change database setting according to your environment.

- Run address_annotator.py to extract information from the postal address.

- Comment and uncomment corresponding line in address_annotator.py to select stacking/ voting integrator.

- Run re_train.py to generate retrain data for CRF model or train stacking integrator.


Created by Zian Zhao June 2018.
