import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import spark.implicits._
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.functions._
val hashSize = 1000 //size of vocabulary hash

/// UDF for Pre-calculating sparse vector norm ///
def calcNorm(vectorA: SparseVector): Double = {
  var norm = 0.0
  for (i <-  vectorA.indices){ norm += vectorA(i)*vectorA(i) }
  (math.sqrt(norm))
}
val calcNormDF = udf[Double,SparseVector](calcNorm)

/// UDF for COSINE SIMILARIITY ////////////
def cosineSimilarity(vectorA: SparseVector, vectorB:SparseVector,normASqrt:Double,normBSqrt:Double) :(Double) = {
  var dotProduct = 0.0
  for (i <-  vectorA.indices){ dotProduct += vectorA(i) * vectorB(i) }
  val div = (normASqrt * normBSqrt)
  if( div == 0 ) (0)
  else (dotProduct / div)
}
val calcCosine = udf[Double,SparseVector,SparseVector,Double,Double](cosineSimilarity)


///////////// TRANSFORM CORPUS & TRAIN TF-IDF //////////////
// load data into DF & tokenize 
val rawdata = sc.textFile("goodbye").map(_.trim.replace("\"", "")).map(_.trim.replace("\'", "")).toDF("rawtext")
val corpus = rawdata.withColumn("text", regexp_replace(rawdata("rawtext"), "[,@&{}$;.!?-]", ""))
val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
val wordsData = tokenizer.transform(corpus)

// create hash and train the IDF model  - the IDF model should be written to file
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(hashsize)
val featurizedData = hashingTF.transform(wordsData)
val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
val idfModel = idf.fit(featurizedData)

// apply TF-IDF to corpus, add norm - the corpus is now ready for comparison
val rescaledData = idfModel.transform(featurizedData)
val normalized = rescaledData.withColumn("norm",calcNormDF(col("features")))
normalized.select("words","features","norm").show()


///////////// FINDING SIMILAR DOCUMENTS //////////////
// read in a new document & format ....
val newrawdata = sc.textFile("goodbye6").map(_.trim.replace("\"", "")).map(_.trim.replace("\'", "")).toDF("raw2").limit(1)
println("This is the text to be matched")
newrawdata.collect.foreach(println)
val newcorpus = newrawdata.withColumn("text", regexp_replace(newrawdata("raw2"), "[,@&{}$;.!?-]", ""))
val newWords = tokenizer.transform(newcorpus)
val newFeature = hashingTF.transform(newWords)
val newRescale = idfModel.transform(newFeature).withColumnRenamed("features", "features2").withColumnRenamed("text", "text2")
val newNormal = newRescale.withColumn("norm2",calcNormDF(col("features2")))
newNormal.select("words","features2","norm2").show()

//// cross join corpus with new text & show top 5 similar ////
val cross = newNormal.crossJoin(normalized)
val cosine = cross.withColumn("similarity",calcCosine(col("features"),col("features2"),col("norm"),col("norm2")))
cosine.sort(desc("similarity")).select("similarity","rawtext").withColumn("similarity",format_number($"similarity",4)).show(5,false)



