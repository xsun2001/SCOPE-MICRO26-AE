package pinn.common

import chisel3._
import circt.stage.ChiselStage
import java.nio.file.{Files, Path, Paths}
import java.util.Comparator
import chisel3.simulator.PeekPokeAPI.TestableData
import scala.jdk.CollectionConverters._
import scala.util.matching.Regex
import scala.util.control.NonFatal

final case class VerilogTarget(path: String, gen: () => RawModule)
final case class GeneratorCliOptions(
    force: Boolean = false,
    filter: Option[Regex] = None,
    dryRun: Boolean = false,
    verbose: Boolean = false
)

object GeneratorCli {
  private val Usage =
    """Options:
      |  --force           Delete the target directory before emitting if it already exists.
      |  --filter <regex>  Only keep targets whose output directory matches the regex.
      |  --dry-run         Print planned actions without deleting or emitting.
      |  --verbose         Print full stacktraces for generator failures.
      |""".stripMargin

  def parse(args: Seq[String]): GeneratorCliOptions = {
    @annotation.tailrec
    def loop(remaining: List[String], options: GeneratorCliOptions): GeneratorCliOptions = remaining match {
      case Nil => options
      case "--force" :: tail =>
        loop(tail, options.copy(force = true))
      case "--dry-run" :: tail =>
        loop(tail, options.copy(dryRun = true))
      case "--verbose" :: tail =>
        loop(tail, options.copy(verbose = true))
      case "--filter" :: Nil =>
        throw new IllegalArgumentException(s"Missing value for --filter\n$Usage")
      case "--filter" :: pattern :: tail =>
        loop(tail, options.copy(filter = Some(compileRegex(pattern))))
      case optionWithValue :: tail if optionWithValue.startsWith("--filter=") =>
        val pattern = optionWithValue.substring("--filter=".length)
        loop(tail, options.copy(filter = Some(compileRegex(pattern))))
      case "--help" :: _ =>
        println(Usage)
        sys.exit(0)
      case unknown :: _ =>
        throw new IllegalArgumentException(s"Unknown generator option: $unknown\n$Usage")
    }

    loop(args.toList, GeneratorCliOptions())
  }

  private def compileRegex(pattern: String): Regex = {
    try {
      pattern.r
    } catch {
      case NonFatal(err) =>
        throw new IllegalArgumentException(s"Invalid regex for --filter: $pattern (${err.getMessage})")
    }
  }
}

object VerilogEmitter {
  val FirtoolOpts: Array[String] = Array(
    "-O=release",
    "-strip-debug-info",
    "-disable-all-randomization",
    "--lowering-options=locationInfoStyle=none",
    "--repl-seq-mem",
    "--repl-seq-mem-file=sram.txt"
  )

  private def isNonEmptyDirectory(outputDir: String): Boolean = {
    val outputPath = Paths.get(outputDir)
    if (!Files.isDirectory(outputPath)) {
      false
    } else {
      val entries = Files.list(outputPath)
      try {
        entries.findFirst().isPresent
      } finally {
        entries.close()
      }
    }
  }

  private def deleteRecursively(path: Path): Unit = {
    if (!Files.exists(path)) {
      return
    }

    val paths = Files.walk(path)
    try {
      paths.sorted(Comparator.reverseOrder()).iterator().asScala.foreach(Files.delete)
    } finally {
      paths.close()
    }
  }

  def emit(gen: => RawModule, outputDir: String, options: GeneratorCliOptions = GeneratorCliOptions()): Unit = {
    val outputPath = Paths.get(outputDir)

    if (options.force && Files.exists(outputPath)) {
      if (options.dryRun) {
        println(s"[DryRun] Would delete $outputDir")
      } else {
        deleteRecursively(outputPath)
        println(s"[Deleted] $outputDir")
      }
    }

    if (options.dryRun) {
      println(s"[DryRun] Would emit $outputDir")
      return
    }

    if (isNonEmptyDirectory(outputDir)) {
      println(s"[Skipped] $outputDir is not empty")
      return
    }

    try {
      Files.createDirectories(outputPath)
      ChiselStage.emitSystemVerilogFile(
        gen,
        args = Array("--target-dir", outputDir, "--split-verilog"),
        firtoolOpts = FirtoolOpts
      )
      println(s"[Emitted] $outputDir")
    } catch {
      case NonFatal(err) =>
        println(s"[Skipped] $outputDir due to error: ${err.getMessage}")
        if (options.verbose) {
          err.printStackTrace(System.out)
        }
    }
  }
}

object GeneratorRunner {
  def run(name: String, targets: Seq[VerilogTarget], options: GeneratorCliOptions): Unit = {
    val selectedTargets = targets.filter { target =>
      options.filter.forall(_.findFirstIn(target.path).isDefined)
    }

    println(s"[$name] Selected ${selectedTargets.size} / ${targets.size} targets")

    selectedTargets.foreach { target =>
      VerilogEmitter.emit(target.gen(), target.path, options)
    }
  }
}

private object GeneratorSweepConfig {
  val TypeList: Seq[ArithTypePair] = Seq(
    (SInt(8.W), SInt(32.W)),
    (SInt(16.W), SInt(32.W)),
    // (Float.fp8_e4m3, Float.fp16),
    // (Float.fp16, Float.fp16),
    (DWFloat.fp8_e4m3, DWFloat.fp16),
    (DWFloat.fp16, DWFloat.fp16)
  )
}

object GenerateNNLut extends App {
  private val options = GeneratorCli.parse(args.toIndexedSeq)
  val dataType: List[ArithType] = List(
    SIntWrapper(32),
    Float.fp16,
    Float.fp32,
    DWFloat.fp16,
    DWFloat.fp32
  )
  val entryCount = List(8, 16, 32)

  val targets = dataType.flatMap { dt =>
    implicit val arithmetic: Arithmetic[dt.T] = dt.arithmetic
    entryCount.map { ec =>
      VerilogTarget(s"generated/nnlut/${dt.name}_$ec", () => new NNLut(ec, dt.dt))
    }
  }

  GeneratorRunner.run("GenerateNNLut", targets, options)
}

object GenerateSystems extends App {
  private val options = GeneratorCli.parse(args.toIndexedSeq)
  // private val arraySizes           = Seq(4, 8, 16, 32, 64, 128)
  private val arraySizes           = Seq(4, 8, 16, 32)
  private val scratchpadDepth      = 1024
  private val accumulatorDepth     = 128
  private val fuseMaxRfEntries     = 10
  private val oneSaSegmentCount    = 4
  private val oneSaSimdWidths      = Seq(4, 16)
  private val pinnacleStripHeights = Seq(2, 4, 8, 16)

  // Interpret the requested fp8 sweep as e4m3 for both HardFloat and DesignWare FP.
  private def buildTypedTargets[T <: Data: Arithmetic](dataType: T, accType: T): Seq[VerilogTarget] = {
    val ev = implicitly[Arithmetic[T]]
    import ev._
    val typeLabel = s"${dataType.name}_${accType.name}"

    val classicalTargets = arraySizes.map { n =>
      VerilogTarget(
        s"generated/systems/classical/n${n}_${typeLabel}",
        () => new pinn.classical.ClassicalSystem(n, dataType, accType, scratchpadDepth, accumulatorDepth)
      )
    }

    val pinnacleTargets = arraySizes.flatMap { n =>
      pinnacleStripHeights.collect {
        case stripHeight if pinn.pinnacle.PinnacleConfig.supports(n, stripHeight) =>
          VerilogTarget(
            s"generated/systems/pinnacle/n${n}_${typeLabel}_h${stripHeight}_c2",
            () =>
              new pinn.pinnacle.PinnacleSystem(
                n,
                dataType,
                accType,
                scratchpadDepth,
                stripHeight = stripHeight,
                accumulatorDepth = accumulatorDepth
              )
          )
      }
    }

    val fuseMaxTargets = arraySizes.map { n =>
      VerilogTarget(
        s"generated/systems/fusemax/n${n}_${typeLabel}",
        () => new pinn.fusemax.FuseMaxSystem(n, dataType, accType, fuseMaxRfEntries, scratchpadDepth)
      )
    }

    val oneSaTargets =
      if (dataType.isFp.isDefined || accType.isFp.isDefined) {
        Seq.empty
      } else {
        arraySizes.flatMap { macEquivalentN =>
          oneSaSimdWidths.flatMap { simdWidth =>
            scaledOneSaMeshSize(macEquivalentN, simdWidth).map { meshN =>
              VerilogTarget(
                s"generated/systems/onesa/n${meshN}_${typeLabel}_simd${simdWidth}",
                () =>
                  new pinn.onesa.OneSaSystem(
                    meshN,
                    dataType.getWidth,
                    accType.getWidth,
                    oneSaSegmentCount,
                    simdWidth = simdWidth,
                    scratchpadDepth = scratchpadDepth,
                    accumulatorDepth = accumulatorDepth
                  )
              )
            }
          }
        }
      }

    val systolicAttentionTargets =
      if (dataType.isFp.isDefined && accType.isFp.isDefined) {
        arraySizes.map { n =>
          VerilogTarget(
            s"generated/systems/systolicattention/n${n}_${typeLabel}",
            () =>
              new pinn.systolicattention.SystolicAttentionSystem(
                n,
                dataType,
                accType,
                scratchpadDepth,
                accumulatorDepth
              )
          )
        }
      } else {
        Seq.empty
      }

    classicalTargets ++ pinnacleTargets ++ oneSaTargets ++ fuseMaxTargets ++ systolicAttentionTargets
  }

  private def scaledOneSaMeshSize(macEquivalentN: Int, simdWidth: Int): Option[Int] = {
    val peScale = math.sqrt(simdWidth.toDouble).toInt
    Option.when(peScale * peScale == simdWidth && macEquivalentN % peScale == 0)(macEquivalentN / peScale)
  }

  val targets = GeneratorSweepConfig.TypeList.flatMap { case tp =>
    implicit val arithmetic: Arithmetic[tp.T] = tp.arithmetic
    buildTypedTargets(tp.dataType, tp.accType)
  }

  GeneratorRunner.run("GenerateSystems", targets, options)
}

object GenerateMeshes extends App {
  private val options             = GeneratorCli.parse(args.toIndexedSeq)
  private val arraySizes          = 4 to 32 by 4
  private val fuseMaxRfEntries    = 10
  private val oneSaMeshSimdWidth  = 2
  private val pinnacleStripCount  = 2

  private def buildTypedTargets[T <: Data: Arithmetic](dataType: T, accType: T): Seq[VerilogTarget] = {
    val ev = implicitly[Arithmetic[T]]
    import ev._
    val typeLabel = s"${dataType.name}_${accType.name}"

    val classicalTargets = arraySizes.map { n =>
      VerilogTarget(
        s"generated/meshes/classical/n${n}_${typeLabel}",
        () => new pinn.classical.SystolicArray(n, dataType, accType)
      )
    }

    val pinnacleTargets = arraySizes.map { n =>
      val stripHeight = n / pinnacleStripCount
      VerilogTarget(
        s"generated/meshes/pinnacle/n${n}_${typeLabel}_h${stripHeight}_c${pinnacleStripCount}",
        () => new pinn.pinnacle.SystolicArray(n, dataType, accType, stripHeight)
      )
    }

    val fuseMaxTargets = arraySizes.map { n =>
      VerilogTarget(
        s"generated/meshes/fusemax/n${n}_${typeLabel}",
        () => new pinn.fusemax.FuseMax2DArray(n, dataType, accType, fuseMaxRfEntries)
      )
    }

    val oneSaTargets = arraySizes.map { n =>
      VerilogTarget(
        s"generated/meshes/onesa/n${n}_${typeLabel}_simd${oneSaMeshSimdWidth}",
        () => new pinn.onesa.OneSaArray(n, oneSaMeshSimdWidth, dataType, accType)
      )
    }

    val systolicAttentionTargets =
      if (dataType.isFp.isDefined && accType.isFp.isDefined) {
        arraySizes.map { n =>
          VerilogTarget(
            s"generated/meshes/systolicattention/n${n}_${typeLabel}",
            () => new pinn.systolicattention.SystolicArray(accType, n, n)
          )
        }
      } else {
        Seq.empty
      }

    classicalTargets ++ pinnacleTargets ++ fuseMaxTargets ++ oneSaTargets ++ systolicAttentionTargets
  }

  val targets = GeneratorSweepConfig.TypeList.flatMap { case tp =>
    implicit val arithmetic: Arithmetic[tp.T] = tp.arithmetic
    buildTypedTargets(tp.dataType, tp.accType)
  }

  GeneratorRunner.run("GenerateMeshes", targets, options)
}
