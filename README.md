===========================================================================================
SENTIMENT ANALYSIS :A Java project to classify movie reviews
==========================================================================================

Project Name: SentimentAnalysis
Package Name: sentimentanalysis
Java Class File:
		nbtrain.java : 	Arguments: 1:
					Path/to/Traindirectory
		nbtest.java  : 	Arguments: 2:
					1. Path/to/modelfile
					2. Path/to/test/directory
		
Libraries referenced:
	None
===============================================================================================
CONFIGURTION INSTRUCTIONS:
===============================================================================================
Environment used to build the project: JDK 1.8 , JAVA v.8
Pre-requisite: Java Development Environment

===============================================================================================
HOW TO RUN THE COMPILED JAR :
===============================================================================================
	To run the Jar file:
		java -jar /path/to/the/jar/nbtrain.jar  Path/to/Traindirectory
		java -jar /path/to/the/jar/nbtest.jar  Path/to/ModelFile Path/to/TestDirectory
	
================================================================================================
ATTACHMENTS:
================================================================================================
1) The working project with source code.
2) 2 jar files  nbtrain.jar
		nbtest.jar
3) 2 tables representing the 
	a. Predictions for dev data
	b. Predictions for test data
4) 2 tables representing the ratio of top 20 terms:
	a. For pos to neg
	b. For neg to pos

================================================================================================
ACCURACY:
================================================================================================
1) Accuracy for "pos" set for dev data: 82%
2) Accuracy for "neg" set for dev data: 66%

================================================================================================
HOW DOES THE PROGRAM WORK?
================================================================================================
1) The program uses Naive Bayes multinomial model to calculate the conditional probabilities
   for the termsoccuring in the training data as provided by the textcat.
2) Laplace smoothing is also implemented in the calculation of the probabilities to cater to the
   zero probability problem, in case the term does not appear in the training doc.
3) The program runs in two folds:
	a. The nbtrain trains the model by reading in the training set.
	b. The nbtest tests the model on the dev data and test data.
4) The input to nbtrain is the training directory, it writes a model file to be used for testing.
5) Input to the nbtest are the model file and the test/dev directory and outputs the predictions
   for the corresponding directory. 6) Log of the probabilities are taken and added together to get the final scores



