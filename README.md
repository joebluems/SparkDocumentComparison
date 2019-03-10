# SparkDocumentComparison

## Background
We wish to search an existing corpus of documents to find potential similarities with a new document. We have chosen to implement in Spark to provide the ability to scale as the potential size and vocabulary of the corpus is significant.<br> 
The eventual target documents are mostly text (i.e. emails), so they need to be tokenized & hashed. We also want to weigh the tokens more heavily for frequent words in a document (TF) and less based on prevalence within the entire corpus (IDF). <br>
To accomplish this, the corpus documents are tokenized, hashed and a TF-IDF model is trained. Both the corpus and new text are parsed and transformed into Sparse Vectors.<br>
For the similarity assessment, the Cosine Similarity between the target text and every corpus document is calculated using the dot product and the norm of each pair. We return the top 5 most similar documents.

## prepare the environment
git clone https://github.com/joebluems/SparkDocumentComparison.git <br>
cd SparkDocumentComparison <br>
(optional) check spark & scala version by running ./spark-shell <br> 
vi build.sbt  ## change spark and scala version if needed <br>
sbt package ## if successful, should create a jar in ./target/scala-2.11 <br>

## notes on the files
./build.sbt - sbt file for spark/scala imports <br>
./goodbye - first 5 chapters of <b>The Long Goodbye</b>, by Raymond Chandler (each sentence is one document in the corpus) <br>
./goodbye6 - chapter 6 of <b>The Long Goodbye</b> <br>
./shell_testing - contains some scala code that could be used in the spark shell for testing <br>
./src/main/scala/Cosine.scala - here's where the code is located <br>

## run the program
<b>Usage: spark-submit --class com.mapr.similarity.Main ./target/scala-2.11/textcomparison_2.11-0.1.0-SNAPSHOT.jar</b> <br>
If you run this without arguments, we assume "goodbye" is the corpus and "goodbye6" will be used as the text to test for similarity against the corpus. <br>
<b> Note #1: if you're running this on a MapR cluster, it will look for the pathname relative to maprfs:// and not be able to find them. Use the arguments with full pathnames instead.</b> <br>
<b> Note #2: hash size of 1,000 is hard-coded - higher hash will have less collisions but require more memory</b> <br>
<b> Note #3: runs in local-mode, which is hard-coded for testing. Switch to cluster or yarn mode if needed. </b> <br>

The output should resemble the following (eventually showing the 5 most similar with the original formatting): <br>
Created TF-IDF on corpus... <br>
This is the text to be matched <br>
<b>First line of the test file to which we want to match similar documents in the corpus</b> <br>
+---+---------+ <br>
sim1   text1... <br>
sim2   text2... <br>
sim3   text3... <br>
sim4   text4... <br>
sim5   text5... <br>
+---+---------+

<b>Usage: spark-submit --class com.mapr.similarity.Main ./target/scala-2.11/textcomparison_2.11-0.1.0-SNAPSHOT.jar  <corpus_file> <test_file></b> <br>
Running with two arguments allows you to specify the location of the corpus and text file to match. <br>

<b>Usage: spark-submit --class com.mapr.similarity.Main ./target/scala-2.11/textcomparison_2.11-0.1.0-SNAPSHOT.jar <corpus_file> <corpus_file></b> <br>
Running with two identical arguments should result in at least one perfect similarity of 1.0000 (this is a way to test) <br>
