name := "ml-class-scala"

scalaVersion := "2.9.1"

libraryDependencies += "org.scalala" %% "scalala" % "1.0.0.RC2-SNAPSHOT"

resolvers ++= Seq(
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo",
  "ondex" at "http://ondex.rothamsted.bbsrc.ac.uk/nexus/content/groups/public"
)

initialCommands := """
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
"""


