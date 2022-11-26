import org.apache.spark.sql.Dataset

object Predictor extends SparkSessionWrapper {

  import org.apache.spark.sql.Encoders
  import spark.implicits._
  import org.apache.spark.ml.regression.LinearRegression

  case class AppVGSale(Rank: Int,
                       Name: String,
                       Platform: String,
                       Year: Int,
                       Genre: String,
                       Publisher: String,
                       NA_Sales: Double,
                       EU_Sales: Double,
                       JP_Sales: Double,
                       Other_Sales: Double,
                       Global_Sales: Double)

  val appVGSalesSchema = Encoders.product[AppVGSale].schema

  def main(args: Array[String]):Unit = {
    val appVGSaleDs: Dataset[AppVGSale] =
      spark
        .read
        .format("csv")
        .option("header", "true")
        .schema(appVGSalesSchema)
        .load("src/main/resources/vgsales.csv")
        .as[AppVGSale]



    val linearRegression = new LinearRegression()
      .setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val linearRegressionModel = linearRegression.fit(appVGSaleDs)

    val summary = linearRegressionModel.summary
    summary.residuals.show()
  }
}
