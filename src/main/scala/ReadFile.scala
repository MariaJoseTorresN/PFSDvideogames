import org.apache.spark.sql.{Dataset, Encoders}

object ReadFile extends  SparkSessionWrapper{
  import spark.implicits._

  case class VGSalesInfo(Rank: Int,
                         Name: String,
                         Platform: String,
                         Year: Long,
                         Genre: String,
                         Publisher: String,
                         NA_Sales: Long,
                         EU_Sales: Long,
                         JP_Sales: Long,
                         Other_Sales: Long,
                         Global_Sales: Long)

  lazy val vgsalesData: Dataset[VGSalesInfo] = spark.read.format("csv")
    .option("header", "true")
    .schema(Encoders.product[VGSalesInfo].schema)
    .load("src/main/resources/vgsales.csv")
    .as[VGSalesInfo]

  def main(args: Array[String]): Unit = {
    vgsalesData.show()
  }
}
