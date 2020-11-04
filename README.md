# Zalo AI Challenge 2020 - Voice Verification


## Description

Voice verification is the process of verifying whether two given utterances belong to a single speaker.
In this challenge, we want you to build a speaker text-independent verification model for Vietnamese voices.

- Input: a pair of utterances. Each utterance's length is from 0.8s-11s.
- Output: Prediction of whether these utterances belong to a single speaker.


## Data

**Training dataset**: The training set should be used to build your models.
For the training set, we provide 400 speakers corresponding to 400 folders.
The folder name has the following format {id-gender-age}.

**Public-test** (public-test.csv): The test set should be used to see how well your model performs on unseen data.
For the public test set, we have 2789 files that make up 50,000 pairs of utterances for evaluation.

- Public-test.csv:  This file contains all test cases of the public test set. Each test case includes a pair of utterances for comparison:
    - 'audio_1': name of the first audio file
    - 'audio_2': name of the second audio file
- Sample_submission.csv: The header is audio_1, audio_2 and label. The first two columns have the same order as in the file pubic-test.csv.
    - label = 1 when these utterances are the same speaker
    - label = 0 when these utterances are not the same speaker

## Author
[KhoiDD](), [ThucTH]()