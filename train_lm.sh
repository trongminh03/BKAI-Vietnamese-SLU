python3 create_kenlm_data.py -j $1

./kenlm/bin/lmplz -o 3 < klm_train.txt > bible.arpa
./kenlm/bin/build_binary bible.arpa your_ngram.binary