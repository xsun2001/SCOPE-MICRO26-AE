ThisBuild / organization := "pinn"
ThisBuild / version := "0.1.0"
ThisBuild / scalaVersion := "2.13.18"

lazy val chiselVersion     = "7.9.0"
lazy val scalaTestVersion  = "3.2.15"

lazy val root = (project in file("."))
  .settings(
    name := "pinn-sa-rtl",
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-language:implicitConversions",
      "-language:reflectiveCalls"
    ),
    Compile / unmanagedSourceDirectories +=
      baseDirectory.value / "third_party" / "berkeley-hardfloat" / "hardfloat" / "src" / "main" / "scala",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.scalatest" %% "scalatest" % scalaTestVersion % Test
    ),
    libraryDependencies += compilerPlugin("org.chipsalliance" % s"chisel-plugin_${scalaVersion.value}" % chiselVersion),
    Test / classLoaderLayeringStrategy := ClassLoaderLayeringStrategy.Flat,
    Test / parallelExecution := false,
    addCommandAlias("genSys", "runMain pinn.common.GenerateSystems"),
    // addCommandAlias("genArrays", "runMain pinn.common.EmitSystolicArrayExamples"),
    // addCommandAlias("genAux", "runMain pinn.common.EmitAuxiliaryModules"),
    // addCommandAlias("genAll", ";genArrays;genAux")
  )
