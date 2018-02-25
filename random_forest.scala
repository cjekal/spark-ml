import scala.io.Source

val csv = Source.fromURL("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data").mkString.split("\\r?\\n")

val rdd = sc.parallelize(csv)

/*
1. Split each line (using “,”) of the CSV file into separate fields
2. Some of the rows in the dataset contain missing values in the seventh field. Remove those.
3. The first column contains an ID. Drop it.
4. Convert all remaining values to floating point numbers
*/
val data = rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

/*
Create a set of LabeledPoint objects that contain a set of feature values and a label (the diagnosis).

The dataset uses 2 to represent a benign diagnosis and 4 to represent a malignant one.
We’ll use 0 and 1, respectively. init returns all the values in a sequence except the
last (which in this case removes the diagnosis value from the list of features).
*/
val labeledPoints = data.map(x => LabeledPoint(if (x.last == 4) 1 else 0, Vectors.dense(x.init)))

/*
Now we split the data into training and test datasets—70% for training and 30% for testing.
*/
val splits = labeledPoints.randomSplit(Array(0.7, 0.3), seed = 5043l)

val trainingData = splits(0)
val testData = splits(1)

/*
Set up the model’s hyperparameters

We have a classification problem, and we’ll use 20 trees each having a maximum
depth of three. The other parameters are pretty standard.
*/
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Gini

val algorithm = Algo.Classification
val impurity = Gini
val maximumDepth = 3
val treeCount = 20
val featureSubsetStrategy = "auto"
val seed = 5043

/*
Train our model!
*/
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.RandomForest

val model = RandomForest.trainClassifier(trainingData, new Strategy(algorithm, impurity, maximumDepth), treeCount, featureSubsetStrategy, seed)

/*
Let’s see how well it works. Using the test dataset we’ll generate a number of
predictions and cross reference them with the actual diagnoses.
*/
val labeledPredictions = testData.map { labeledPoint =>
  val predictions = model.predict(labeledPoint.features)
  (labeledPoint.label, predictions)
}

/*
Let’s look at the precision of the classifier—i.e. how many predictions it gets right.
*/
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val evaluationMetrics = new MulticlassMetrics(labeledPredictions.map(x => (x._1, x._2)))

evaluationMetrics.precision

/*
What about the confusion matrix?

true positive   false positive
122.0           2.0
6.0             77.0
false negative  true negative
*/
evaluationMetrics.confusionMatrix
