import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val randomSeed = 1234

val data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/tmp/data/ml.shuffled.csv.bz2").filter("`confirmedMatchesAllegedInjury` is not null and `i:1` is not null and `i:27` is not null").cache()

// val labelIndexer = new StringIndexer().setInputCol("HasQAFailure").setOutputCol("indexedHasQAFailure").fit(data)
val ignoreColumns = "id:AuditTicket,id:Review,id:Claim,id:ClaimSchema,id:Version,id:Exposed,timestamp:AuditStarted,timestamp:Reviewed".split(",")
val toDropColumns = "cnt:TotalPagesOfDocuments,cnt:SuccessfulClaimsForFirm,cnt:UnsuccessfulClaimsForFirm,ratio:SuccessfulClaimsRatioForFirm,cnt:SuccessfulClaimsForAttorney,cnt:UnsuccessfulClaimsForAttorney,ratio:SuccessfulClaimsRatioForAttorney,cnt:MUTs,cnt:ExposureOverrides".split(",")
val labelColumns = "HasQAFailure".split(",")
val featureCols = data.columns.diff(labelColumns).diff(ignoreColumns).diff(toDropColumns)
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
assembler.transform(data)

val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), randomSeed)

val rf = new RandomForestClassifier().setLabelCol("HasQAFailure").setFeaturesCol("features")
rf.setFeatureSubsetStrategy("auto")
rf.setSeed(randomSeed)

// val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

val pipeline = new Pipeline().setStages(Array(assembler, rf))

val paramGridBuilder = new ParamGridBuilder()
paramGridBuilder.addGrid(rf.impurity, Array("Gini", "entropy"))
paramGridBuilder.addGrid(rf.maxDepth, Array(3, 5, 7, 9, 11, 13))
paramGridBuilder.addGrid(rf.numTrees, Array(20, 200, 400, 600, 800, 1000))
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
