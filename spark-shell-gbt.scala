import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

val randomSeed = 1234

val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/tmp/data/ml.shuffled.cleaned.truncated.csv")

val labelIndexer = new StringIndexer().setInputCol("HasQAFailure").setOutputCol("indexedHasQAFailure").fit(data)
val ignoreColumns = "id:AuditTicket,id:Review,id:Claim,id:ClaimSchema,id:Version,id:Exposed,timestamp:AuditStarted,timestamp:Reviewed".split(",")
val toDropColumns = "cnt:TotalPagesOfDocuments,cnt:SuccessfulClaimsForFirm,cnt:UnsuccessfulClaimsForFirm,ratio:SuccessfulClaimsRatioForFirm,cnt:SuccessfulClaimsForAttorney,cnt:UnsuccessfulClaimsForAttorney,ratio:SuccessfulClaimsRatioForAttorney,cnt:MUTs,cnt:ExposureOverrides".split(",")
val labelColumns = "HasQAFailure".split(",")
val featureCols = data.columns.diff(labelColumns).diff(ignoreColumns).diff(toDropColumns)
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
assembler.transform(data)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), randomSeed)

val gbt = new GBTClassifier().setLabelCol("indexedHasQAFailure").setFeaturesCol("features")
gbt.setMaxIter(10)
//gbt.setLossType("logistic")
// gbt.setMaxBins(0)
gbt.setMaxDepth(3)
gbt.setMinInfoGain(0.01)
gbt.setMinInstancesPerNode(10)
//gbt.setStepSize(0.0)
//gbt.setSubsamplingRate(0.0)
//gbt.setThresholds(Array())
gbt.setSeed(randomSeed)

val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(labelIndexer, assembler, gbt, labelConverter))

val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "HasQAFailure", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedHasQAFailure").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println("Learned classification GBT model:\n" + gbtModel.toDebugString)

:quit


/* this is all grid-searching
val paramGridBuilder = new ParamGridBuilder()
paramGridBuilder.addGrid(rf.impurity, Array("Entropy", "Gini"))
paramGridBuilder.addGrid(rf.maxDepth, Array(8, 10, 12, 20, 30, 50))
paramGridBuilder.addGrid(rf.numTrees, Array(200, 300, 600))
val paramGrid = paramGridBuilder.build()

val cv = new CrossValidator()
cv.setEstimator(pipeline)
cv.setEvaluator(new BinaryClassificationEvaluator().setLabelCol("HasQAFailure").setMetricName("areaUnderROC"))
cv.setEstimatorParamMaps(paramGrid)
cv.setNumFolds(3)

val cvModel = cv.fit(trainingData)
cvModel.bestModel.asInstanceOf[PipelineModel].stages(1).extractParamMap

//val model = pipeline.fit(trainingData)

val predictions = cvModel.bestModel.transform(testData)

val accuracyEvaluator = new MulticlassClassificationEvaluator().setLabelCol("HasQAFailure").setPredictionCol("prediction")
val accuracy = accuracyEvaluator.setMetricName("accuracy").evaluate(predictions)

val evaluator = new BinaryClassificationEvaluator().setLabelCol("HasQAFailure").setRawPredictionCol("rawPrediction")
val areaUnderROC = evaluator.setMetricName("areaUnderROC").evaluate(predictions)

val areaUnderPR = evaluator.setMetricName("areaUnderPR").evaluate(predictions)
println(s"Accuracy = $accuracy, Area Under ROC = $areaUnderROC, Area Under PR = $areaUnderPR")

// val rfModel = model.stages(1).asInstanceOf[RandomForestClassificationModel]
// featureCols.zip(rfModel.featureImportances.toArray).sortBy(-_._2)
// println("Learned classification forest model:\n" + rfModel.toDebugString)
*/
