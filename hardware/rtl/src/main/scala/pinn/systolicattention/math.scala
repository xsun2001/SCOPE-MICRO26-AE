package pinn.systolicattention

import chisel3._
import chisel3.util._
import pinn.common._

object FpBits {
  private def float32Bits(value: Double): BigInt =
    BigInt(java.lang.Float.floatToIntBits(value.toFloat) & 0xffffffffL)

  private def float16Bits(value: Double): BigInt = {
    val f    = java.lang.Float.floatToIntBits(value.toFloat)
    val sign = (f >>> 16) & 0x8000
    val exp  = (f >>> 23) & 0xff
    val mant = f & 0x7fffff

    if (exp == 0xff) {
      val nanMant = if (mant == 0) 0 else 0x200
      return BigInt(sign | 0x7c00 | nanMant)
    }

    val halfExp = exp - 127 + 15
    if (halfExp >= 0x1f) {
      BigInt(sign | 0x7c00)
    } else if (halfExp <= 0) {
      if (halfExp < -10) {
        BigInt(sign)
      } else {
        val mantWithHidden = mant | 0x800000
        val shift          = 14 - halfExp
        val rounded = {
          val base    = mantWithHidden >>> shift
          val rem     = mantWithHidden & ((1 << shift) - 1)
          val half    = 1 << (shift - 1)
          val roundUp = rem > half || (rem == half && (base & 1) == 1)
          if (roundUp) base + 1 else base
        }
        BigInt(sign | rounded)
      }
    } else {
      val mantRounded = {
        val base    = mant >>> 13
        val rem     = mant & 0x1fff
        val half    = 0x1000
        val roundUp = rem > half || (rem == half && (base & 1) == 1)
        if (roundUp) base + 1 else base
      }
      if (mantRounded == 0x400) {
        BigInt(sign | ((halfExp + 1) << 10))
      } else {
        BigInt(sign | (halfExp << 10) | (mantRounded & 0x3ff))
      }
    }
  }

  def bits(value: Double, expWidth: Int, sigWidth: Int): BigInt = (expWidth, sigWidth) match {
    case (8, 24) => float32Bits(value)
    case (5, 11) => float16Bits(value)
    case other =>
      throw new IllegalArgumentException(s"Unsupported floating-point format for constant generation: $other")
  }

  def bits[T <: Data: Arithmetic](value: Double, fmt: T): BigInt = {
    val ev = implicitly[Arithmetic[T]]
    import ev._

    fmt.isFp match {
      case Some((expWidth, sigWidth)) => bits(value, expWidth, sigWidth)
      case None =>
        throw new IllegalArgumentException(
          s"SystolicAttention constant generation requires floating-point Arithmetic, got ${fmt.getClass.getSimpleName}"
        )
    }
  }
}

object SystolicAttentionPwl {
  def pieceCount = 8

  private def pieceBounds(idx: Int, pieces: Int): (Double, Double) = {
    val hi = -idx.toDouble / pieces.toDouble
    val lo = -(idx + 1).toDouble / pieces.toDouble
    (lo, hi)
  }

  def rawSlopes[T <: Data: Arithmetic](fmt: T, pieces: Int = pieceCount): Seq[BigInt] =
    Seq.tabulate(pieces) { idx =>
      val (lo, hi) = pieceBounds(idx, pieces)
      val yLo      = math.pow(2.0, lo)
      val yHi      = math.pow(2.0, hi)
      val slope    = (yHi - yLo) / (hi - lo)
      FpBits.bits(slope, fmt)
    }

  def rawIntercepts[T <: Data: Arithmetic](fmt: T, pieces: Int = pieceCount): Seq[BigInt] =
    Seq.tabulate(pieces) { idx =>
      val (_, hi) = pieceBounds(idx, pieces)
      val slope = {
        val (loB, hiB) = pieceBounds(idx, pieces)
        val yLo        = math.pow(2.0, loB)
        val yHi        = math.pow(2.0, hiB)
        (yHi - yLo) / (hiB - loB)
      }
      val yHi       = math.pow(2.0, hi)
      val intercept = yHi - slope * hi
      FpBits.bits(intercept, fmt)
    }

  def encodedIntercepts[T <: Data: Arithmetic](fmt: T, pieces: Int = pieceCount): Seq[BigInt] = {
    val ev = implicitly[Arithmetic[T]]
    import ev._

    val (_, sigWidth) = fmt.isFp.getOrElse(
      throw new IllegalArgumentException("encoded intercept generation requires floating-point Arithmetic")
    )
    val fracWidth     = sigWidth - 1
    val expShift      = fracWidth
    val (expWidth, _) = fmt.isFp.get
    val expMask       = ((BigInt(1) << expWidth) - 1) << expShift

    rawIntercepts(fmt, pieces).zipWithIndex.map { case (raw, idx) =>
      val rawExp     = (raw & expMask) >> expShift
      val expLsb     = rawExp & 1
      val encodedExp = (BigInt(idx) << 1) | expLsb
      (raw & ~expMask) | (encodedExp << expShift)
    }
  }

  def attentionScaleBits[T <: Data: Arithmetic](fmt: T, dk: Int): BigInt =
    FpBits.bits(math.log(math.E) / math.log(2.0) / math.sqrt(dk.toDouble), fmt)

  def oneBits[T <: Data: Arithmetic](fmt: T): BigInt  = FpBits.bits(1.0, fmt)
  def zeroBits[T <: Data: Arithmetic](fmt: T): BigInt = FpBits.bits(0.0, fmt)
}

object SystolicAttentionScaleCmd extends ChiselEnum {
  val Idle, Mul, Fma, Exp2 = Value
}

object SystolicAttentionScale {
  def applyPow2Integer[T <: Data: Arithmetic](value: T, intPart: SInt): T = {
    val ev = implicitly[Arithmetic[T]]
    import ev._

    val (expWidth, sigWidth) = value.isFp.getOrElse(
      throw new IllegalArgumentException(
        s"SystolicAttentionScale requires floating-point Arithmetic, got ${value.getClass.getSimpleName}"
      )
    )

    val out       = Wire(chiselTypeOf(value))
    val raw       = value.asUInt
    val sign      = raw(value.getWidth - 1)
    val fracWidth = sigWidth - 1
    val expHi     = value.getWidth - 2
    val expLo     = fracWidth
    val expField  = raw(expHi, expLo)
    val fracField = raw(fracWidth - 1, 0)

    val isZeroOrSubnormal = expField === 0.U
    val isSpecial         = expField.andR
    val adjustedExp       = expField.zext + intPart
    val underflow         = adjustedExp <= 0.S
    val overflow          = adjustedExp >= ((1 << expWidth) - 1).S(adjustedExp.getWidth.W)

    out := value
    when(isZeroOrSubnormal || isSpecial) {
      out := value
    }.elsewhen(underflow) {
      out := Cat(sign, 0.U((value.getWidth - 1).W)).asTypeOf(chiselTypeOf(value))
    }.elsewhen(overflow) {
      out := Cat(sign, Fill(expWidth, 1.U(1.W)), 0.U(fracWidth.W)).asTypeOf(chiselTypeOf(value))
    }.otherwise {
      out := Cat(sign, adjustedExp.asUInt(expWidth - 1, 0), fracField).asTypeOf(chiselTypeOf(value))
    }

    out
  }
}

class SystolicAttentionScale[T <: Data: Arithmetic](fmt: T, pieces: Int = SystolicAttentionPwl.pieceCount)
    extends Module {
  val io = IO(new Bundle {
    val cmd          = Input(SystolicAttentionScaleCmd())
    val x            = Input(fmt.cloneType)
    val w            = Input(fmt.cloneType)
    val b            = Input(fmt.cloneType)
    val coeffEncoded = Input(Bool())
    val expIdx       = Output(UInt(math.max(1, log2Ceil(pieces)).W))
    val expMatch     = Output(Bool())
    val out          = Output(fmt.cloneType)
  })

  val ev = implicitly[Arithmetic[T]]
  import ev._

  require(
    fmt.isFp.isDefined,
    s"SystolicAttentionScale requires floating-point Arithmetic, got ${fmt.getClass.getSimpleName}"
  )

  private val (expWidth, sigWidth) = fmt.isFp.get
  private val bias                 = (1 << (expWidth - 1)) - 1
  private val zero                 = fmt.zero
  private val pieceBits            = math.max(1, log2Ceil(pieces))

  private val split   = io.x.splitForExp.get
  private val fracMag = split._2
  private val fracFp  = split._3

  private val expIdx =
    if (pieceBits >= fracMag.getWidth) 0.U(pieceBits.W)
    else fracMag(fracMag.getWidth - 1, fracMag.getWidth - pieceBits)
  private val expectedPiece =
    if (pieceBits >= expWidth - 1) expIdx(expWidth - 2, 0)
    else Cat(0.U((expWidth - 1 - pieceBits).W), expIdx)

  private val rawB            = io.b.asUInt
  private val encodedExp      = rawB(fmt.getWidth - 2, sigWidth - 1)
  private val encodedPiece    = encodedExp(expWidth - 1, 1)
  private val encodedExpLsb   = encodedExp(0)
  private val restoredExp     = Cat((bias >> 1).U((expWidth - 1).W), encodedExpLsb)
  private val restoredEncoded = Cat(rawB(fmt.getWidth - 1), restoredExp, rawB(sigWidth - 2, 0)).asTypeOf(fmt.cloneType)
  private val expAddend       = Mux(io.coeffEncoded, restoredEncoded, io.b)

  private val macAddend = Wire(fmt.cloneType)
  private val macWeight = Wire(fmt.cloneType)
  private val macInput  = Wire(fmt.cloneType)

  macAddend := zero
  macWeight := io.w
  macInput  := io.x

  switch(io.cmd) {
    is(SystolicAttentionScaleCmd.Fma) {
      macAddend := io.b
    }
    is(SystolicAttentionScaleCmd.Exp2) {
      macAddend := expAddend
      macWeight := io.w
      macInput  := fracFp
    }
  }

  private val macOut = macAddend.mac(macWeight, macInput)

  io.expIdx   := expIdx
  io.expMatch := !io.coeffEncoded || encodedPiece === expectedPiece
  io.out := MuxLookup(io.cmd.asUInt, zero)(
    Seq(
      SystolicAttentionScaleCmd.Mul.asUInt  -> macOut,
      SystolicAttentionScaleCmd.Fma.asUInt  -> macOut,
      SystolicAttentionScaleCmd.Exp2.asUInt -> SystolicAttentionScale.applyPow2Integer(macOut, split._1)
    )
  )
}

class DirectExp2Pwl[T <: Data: Arithmetic](fmt: T, pieces: Int = SystolicAttentionPwl.pieceCount) extends Module {
  val io = IO(new Bundle {
    val in  = Input(fmt.cloneType)
    val out = Output(fmt.cloneType)
  })

  val ev = implicitly[Arithmetic[T]]
  import ev._

  require(
    fmt.isFp.isDefined,
    s"DirectExp2Pwl requires floating-point Arithmetic, got ${fmt.getClass.getSimpleName}"
  )

  private val slopes =
    VecInit(SystolicAttentionPwl.rawSlopes(fmt, pieces).map(_.U(fmt.getWidth.W).asTypeOf(fmt.cloneType)))
  private val intercepts =
    VecInit(SystolicAttentionPwl.rawIntercepts(fmt, pieces).map(_.U(fmt.getWidth.W).asTypeOf(fmt.cloneType)))
  private val scale = Module(new SystolicAttentionScale(fmt, pieces))
  private val idx   = scale.io.expIdx

  scale.io.cmd          := SystolicAttentionScaleCmd.Exp2
  scale.io.x            := io.in
  scale.io.w            := Mux1H(UIntToOH(idx, pieces), slopes)
  scale.io.b            := Mux1H(UIntToOH(idx, pieces), intercepts)
  scale.io.coeffEncoded := false.B

  io.out := scale.io.out
}
