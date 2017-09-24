The video_to_text showcase task testing set consists of the following files:

1- vines.url.testingSet
2- vines.textDescription.A.testingSet
3- vines.textDescription.B.testingSet

In each of the above 3 files, the first column represents
the ID of a vine URL or a text description.

To submit results for the subtask of "Matching and Ranking":
-----------------------------------------------------------

Return for each video URL a ranked list of the most likely text 
description that correspond (was annotated) to the video from 
each of the sets A and B. Please use the following format in your 
submission files:

rank URL_ID TextDescription_ID

where:
rank - Is an integer number represents the likelihood that the "textual description" represented by TextDescription_ID 
       taken from vines.textDescription.A.testingSet or vines.textDescription.B.testingSet most likely describes the video URL.
       (the lower rank numbers, the higher the confidence/rank).

URL_ID - Is the URL id taken from the first column in the file vines.url.testingSet

TextDescription_ID - Is the textual description id taken from the first column in the files vines.textDescription.A.testingSet or vines.textDescription.B.testingSet

Please submit different run files for each of the textual description sets A and B

Example of a snippet from a run file:
1 1 367
2 1 78
3 1 1289
.
.
1 2 278
2 2 902
.
.
1915 1915 10
       
To submit results for the subtask of "Description Generation":
-------------------------------------------------------------

Automatically generate for each video URL a text description (1 sentence) 
independently and without taking into consideration the existence of sets A and B

Please use the following format in your submission files:

URL_ID TextDescription

Where:
URL_ID - Is the URL id taken from the first column in the file vines.url.testingSet

TextDescription - Is the system generated 1 sentence text description.

Example of a snippet from a run file:
10 a man and a woman riding in a car at	night


Notes:
- Systems are allowed to submit up to 4 runs for each set in the Matching and Ranking subtask 
  and 4 runs in the Description Generation subtask.
- Please use the strings ".A." and ".B." as part of your run file names to diffrentiate between set A and set B run files.
- A run should include results for all the testing video URLS (no missing video URL_ID will be allowed).
- No duplicate result pairs of <rank> AND <URL_ID> are allowed (please submit only 1 unique set of ranks per URL_ID).
- All automatic text descriptions should be in English.
