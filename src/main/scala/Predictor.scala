import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.{Dataset, Encoders}

object Predictor extends SparkSessionWrapper {

  import spark.implicits._
  import org.apache.spark.ml.regression.LinearRegression

  case class VGSalesInfo(Rank: Int,
                         Name: String,
                         Platform: String,
                         Year: Long,
                         Genre: String,
                         Publisher: String,
                         NA_Sales: Double,
                         EU_Sales: Double,
                         JP_Sales: Double,
                         Other_Sales: Double,
                         Global_Sales: Double)

  lazy val vgsalesData: Dataset[VGSalesInfo] = spark.read.format("csv")
    .option("header", "true")
    .schema(Encoders.product[VGSalesInfo].schema)
    .load("src/main/resources/vgsales.csv")
    .as[VGSalesInfo]

  var vgsalesDF = vgsalesData.toDF()

  def indexador(inputCol: String, outputCol: String): StringIndexer = {
    new StringIndexer().setInputCol(inputCol).setOutputCol(outputCol)
  }

  def main(args: Array[String]): Unit = {
    //vgsalesData.show()
    val platformIndexed = indexador("Platform","PlatformIndex")
    vgsalesDF = platformIndexed.fit(vgsalesDF).transform(vgsalesDF)
    val genreIndexed = indexador("Genre", "GenreIndex")
    vgsalesDF = genreIndexed.fit(vgsalesDF).transform(vgsalesDF)
    val publisherIndexed = indexador("Publisher", "PublisherIndex")
    vgsalesDF = publisherIndexed.fit(vgsalesDF).transform(vgsalesDF)

    //vgsalesDF.show()
    val dataFrame2 = vgsalesDF.drop("Rank", "Name", "Year")
    val dataFrame3 = vgsalesDF.drop("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales")

    val columns = Array("PlatformIndex", "GenreIndex", "PublisherIndex", "NA_Sales", "EU_Sales")
    val assembler = new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")

    val output = assembler.transform(dataFrame2)
    //output.show()

    val Array(training, test) = output.randomSplit(Array(0.7, 0.3), 18)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)
    val scalerModel = scaler.fit(training)
    val scaledData = scalerModel.transform(training)
    //scaledData.show()

    val scalerModelTest = scaler.fit(test)
    val scaledDataTest = scalerModelTest.transform(test)
    //scaledDataTest.show()

    val linearRegression = new LinearRegression()

    linearRegression.setLabelCol("Global_Sales")
    val linearRegressionModel = linearRegression.fit(scaledData)
    val linearRegressionPredictions = linearRegressionModel.transform(scaledDataTest)

    linearRegressionPredictions.show()
    linearRegressionModel.summary.residuals.show()

    val RMSE = linearRegressionModel.summary.rootMeanSquaredError
    println(s"Distancia media cuadratica minima: ${RMSE}")
    val exactitud = linearRegressionModel.summary.r2
    println(s"Exactitud:  ${exactitud*100}%")
    val i = linearRegressionModel.intercept
    val c = linearRegressionModel.coefficients(0)
    print(s"Modelo matematico: Y = ${c} * X + ${i}")

  }
}
