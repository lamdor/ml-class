package ex1
import scalala.scalar._
import scalala.tensor.::
import scalala.tensor.mutable._
import scalala.tensor.dense._
import scalala.tensor.sparse._
import scalala.library.Library._
import scalala.library.LinearAlgebra._
import scalala.library.Statistics._
import scalala.library.Plotting._
import scalala.operators.Implicits._
import scala.annotation.tailrec

object WarmUpExercise {
  def apply() = {
    DenseMatrix.eye[Int](5)
  }
}

object LinearRegressionSingleData {
  val dataLines = io.Source.fromFile("ex1data1.txt").getLines.toSeq
  val data = DenseMatrix(dataLines.map(_.split(',').map(_.toDouble)): _*)
  val X = DenseMatrix.horzcat(DenseMatrix.ones[Double](data.numRows, 1),
                              DenseMatrix(data(::, 0).t).t)
  val y = data(::, 1)
  val m = y.length
}  
  
object PlottingTheData {
  import LinearRegressionSingleData._
  
  def apply() = {
    plot(X(::, 1), y, '.')
    ylabel("Profit in $10,000s")
    xlabel("Population of City in 10,000s")
  }
}

object ComputeCost {
  def apply(x: DenseMatrix[Double],
            y: DenseVector[Double],
            theta: DenseVector[Double]): Double = {
    val m = y.length
    val predictions = x * theta.asCol
    val sqrErrors = (predictions :- y) :^ 2
    (1.0 / (2 * m)) * sqrErrors.sum
  }
}

object RunComputeCostForSingleValues {
  import LinearRegressionSingleData._
  
  def apply() = {
    ComputeCost(X, y, DenseVector(0, 0))
  }
}

object GradientDescent {
   def apply(x: DenseMatrix[Double],
             y: DenseVector[Double],
             theta: DenseVector[Double],
             alpha: Double              = 0.01,
             numIters: Int              = 1500): (DenseVector[Double], DenseVector[Double]) = {
     val m = y.length
     
     @tailrec def run(iters:Int,
                      theta: DenseVector[Double],
                      history: DenseVector[Double]): (DenseVector[Double], DenseVector[Double]) =
       if (iters == 0) {
         (theta, history) 
       } else {
         val predictions = x * theta.asCol
         val newTheta = theta - ((x.t * (predictions :- y)) :* (alpha / m))
         val cost = DenseVector(ComputeCost(x, y, theta))
         val newHistory = if (history.size == 0)
           cost
         else
           DenseVector.vertcat(history, cost)
         run(iters - 1, newTheta, newHistory)
       }
     
     run(numIters, theta, DenseVector[Double]())
   }
 }

object RunGradientDescentForSingleValues {
  import LinearRegressionSingleData._
  
  def apply() = {
    val (theta, jHist) = GradientDescent(X, y, DenseVector.zeros(X.numCols))
    plot.hold = true
    plot(X(::, 1), y, '.')
    ylabel("Profit in $10,000s")
    xlabel("Population of City in 10,000s")
    plot(X(::, 1), X * theta.asCol, '-')
    val predict1 = DenseVector(1, 3.5).asRow * theta.asCol
    printf("For population = 35,000, we predict a profit of %f\n",
            predict1 * 10000);
    val predict2 = DenseVector(1, 7).asRow * theta.asCol
    printf("For population = 70,000, we predict a profit of %f\n",
            predict2 * 10000);

    // legend("Training data", "Linear regression")
  }
}

object LinearRegressionMultiData {
  val dataLines = io.Source.fromFile("ex1data2.txt").getLines.toSeq
  val data = DenseMatrix(dataLines.map(_.split(',').map(_.toDouble)): _*)
  val X = data(::, 0 to 1).toDense
  val y = data(::, 2)
  val m = y.length
}  

object FeatureNormalize {
  def apply(x: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double]) = {
    val normalized = x.copy
    val muAndSigmas = (0 until normalized.numCols) map { i => 
      val feature = normalized(::, i)
      val mu = mean(feature)
      val sigma = stddev(feature)
      normalized(::, i) := (feature :- mu) :/ sigma
      (mu, sigma)
    }
    val mu = DenseVector(muAndSigmas.map(_._1): _*)
    val sigma = DenseVector(muAndSigmas.map(_._2): _*)
    (normalized, mu, sigma)
  }
}

object RunGradientDescentForMultiValues {
  import LinearRegressionMultiData._
  
  def apply() = {
    val (xNorm, mu, sigma) = FeatureNormalize(X)
    // ComputeCost(xNorm, y, DenseVector(0, 0, 0))
    val xNormWithOnes = DenseMatrix.horzcat(DenseMatrix.ones[Double](xNorm.numRows, 1),
                                            xNorm)
    val (theta, jHist) = GradientDescent(xNormWithOnes,
                                         y,
                                         DenseVector.zeros(xNormWithOnes.numCols),
                                         0.03,
                                         5000)
    val house = DenseVector(1650, 3)
    val normalized = DenseVector.vertcat(DenseVector(1.0),
                                         (house :- mu) :/ sigma)
    val predict = normalized.asRow * theta.asCol
    printf("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n", predict);
  }
}

object NormalEquation {
  def apply(x: DenseMatrix[Double],
            y: DenseVector[Double]): DenseVector[Double] = {
    pinv((x.t * x).toDense) * x.t * y.asCol
  }
}  
  
object RunNormalEquationForMultiValues {
  import LinearRegressionMultiData._
  
  def apply() = {
    val xWithOnes = DenseMatrix.horzcat(DenseMatrix.ones[Double](m, 1), X)
    val theta = NormalEquation(xWithOnes, y)
    
    val house = DenseVector(1.0, 1650.0, 3.0)
    val predict = house.asRow * theta.asCol
    printf("Predicted price of a 1650 sq-ft, 3 br house (using normal equation):\n $%f\n", predict);
  }
}
