# run the kiwi sample in stand alone
python kiwi_rail.py

#test

# test 2 on training branch

#how to branch

# create a branch called training from master

# checkout master
git checkout master

# create a branch called training from master
git checkout -b training master

# switch to training branch
git checkout training

# create a branch called melika from training branch
git checkout -b melika training

master -> |
           training -> |
                       melika # write code here
                       then stage, message, commit, push



