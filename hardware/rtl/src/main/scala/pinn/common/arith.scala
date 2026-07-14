package pinn.common

import chisel3._
import chisel3.util._
import hardfloat._

sealed trait ArithType {
  type T <: Data
  val dt: T
  implicit val arithmetic: Arithmetic[T]
  val name: String
}

object ArithType {
  def apply[U <: Data](dtValue: U)(implicit arith: Arithmetic[U]): ArithType = new ArithType {
    override type T = U
    override val dt: U                              = dtValue
    override implicit val arithmetic: Arithmetic[U] = arith
    override val name: String = {
      import arithmetic._
      dt.name
    }
  }

  implicit def fromData[U <: Data](dataType: U)(implicit arith: Arithmetic[U]): ArithType =
    apply(dataType)
}

sealed trait ArithTypePair {
  type T <: Data
  val dataType: T
  val accType: T
  implicit val arithmetic: Arithmetic[T]
}

object ArithTypePair {
  def apply[U <: Data](dtValue: U, accValue: U)(implicit arith: Arithmetic[U]): ArithTypePair = new ArithTypePair {
    override type T = U
    override val dataType: U                        = dtValue
    override val accType: U                         = accValue
    override implicit val arithmetic: Arithmetic[U] = arith
  }

  implicit def fromData[U <: Data](typePair: (U, U))(implicit arith: Arithmetic[U]): ArithTypePair =
    apply(typePair._1, typePair._2)
}

case class Float(expWidth: Int, sigWidth: Int) extends Bundle {
  val bits = UInt((expWidth + sigWidth).W)

  val bias: Int = (1 << (expWidth - 1)) - 1
}

object Float {
  def fp16     = Float(5, 11)
  def bf16     = Float(8, 7)
  def fp32     = Float(8, 24)
  def fp8_e4m3 = Float(5, 3)
  def fp8_e5m2 = Float(6, 2)
}

case class DWFloat(expWidth: Int, sigWidth: Int) extends Bundle {
  val bits = UInt((expWidth + sigWidth).W)

  val bias: Int = (1 << (expWidth - 1)) - 1
}

object DWFloat {
  def fp16     = DWFloat(5, 11)
  def bf16     = DWFloat(8, 7)
  def fp32     = DWFloat(8, 24)
  def fp8_e4m3 = DWFloat(5, 3)
  def fp8_e5m2 = DWFloat(6, 2)
}

case class SIntWrapper(w: Int) extends Bundle {
  val bits = UInt(w.W)
}

object SIntWrapper {
  def apply(x: SInt): SIntWrapper = SIntWrapper(x.asUInt)

  def apply(x: UInt): SIntWrapper = {
    val w       = x.getWidth
    val wrapper = Wire(SIntWrapper(w))
    wrapper.bits := x
    wrapper
  }
}

// The Arithmetic typeclass which implements various arithmetic operations on custom datatypes
abstract class Arithmetic[T <: Data] {
  implicit def cast(t: T): ArithmeticOps[T]
}

final case class ValidDivResult[T <: Data](inReady: Bool, out: ValidIO[T])

abstract class ArithmeticOps[T <: Data](self: T) {
  def name: String

  def *(t: T): T
  def /(t: T): T
  def mac(m1: T, m2: T): T // Returns (m1 * m2 + self)
  def +(t: T): T
  def -(t: T): T
  def >>(u: UInt): T // This is a rounding shift! Rounds away from 0
  def >(t: T): Bool
  def identity: T
  def withWidthOf(t: T): T
  def clippedToWidthOf(t: T): T // Like "withWidthOf", except that it saturates
  def relu: T
  def zero: T
  def minimum: T
  def abs: T
  def neg: T

  // Split a non-positive fp number into its integer part (SInt), fractional part in fixed point (UInt) and in fp (T)
  def splitForExp: Option[(SInt, UInt, T)] = None
  // If is fp, return (expWidth, sigWidth)
  def isFp: Option[(Int, Int)] = None
  // Optional valid-producing divider. Callers must keep request operands stable until `inReady` is high.
  def divWithValid(t: T, inValid: Bool): Option[ValidDivResult[T]] = None

  // Optional parameters, which only need to be defined if you want to enable various optimizations for transformers
  // def divider(denom_t: UInt, options: Int = 0): Option[(DecoupledIO[UInt], DecoupledIO[T])]      = None
  // def sqrt: Option[(DecoupledIO[UInt], DecoupledIO[T])]                                          = None
  // def reciprocal[U <: Data](u: U, options: Int = 0): Option[(DecoupledIO[UInt], DecoupledIO[U])] = None
  // def mult_with_reciprocal[U <: Data](reciprocal: U)                                             = self
}

object Arithmetic {
  implicit object UIntArithmetic extends Arithmetic[UInt] {
    override implicit def cast(self: UInt): ArithmeticOps[UInt] = new ArithmeticOps(self) {

      override def name                    = s"u${self.getWidth}"
      override def *(t: UInt)              = self * t
      override def /(t: UInt)              = Mux(t === 0.U, 0.U.asTypeOf(self), self / t)
      override def mac(m1: UInt, m2: UInt) = m1 * m2 + self
      override def +(t: UInt)              = self + t
      override def -(t: UInt)              = self - t

      override def >>(u: UInt) = {
        // The equation we use can be found here: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm

        // TODO Do we need to explicitly handle the cases where "u" is a small number (like 0)? What is the default behavior here?
        val point_five = Mux(u === 0.U, 0.U, self(u - 1.U))
        val zeros      = Mux(u <= 1.U, 0.U, self.asUInt & ((1.U << (u - 1.U)).asUInt - 1.U)) =/= 0.U
        val ones_digit = self(u)

        val r = point_five & (zeros | ones_digit)

        (self >> u).asUInt + r
      }

      override def >(t: UInt): Bool = self > t

      override def withWidthOf(t: UInt) = self.asTypeOf(t)

      override def clippedToWidthOf(t: UInt) = {
        val sat = ((1 << (t.getWidth - 1)) - 1).U
        Mux(self > sat, sat, self)(t.getWidth - 1, 0)
      }

      override def relu: UInt = self

      override def zero: UInt     = 0.U
      override def identity: UInt = 1.U
      override def minimum: UInt  = 0.U
      override def abs: UInt      = self
      override def neg: UInt      = self
    }
  }

  implicit object SIntWrapperArithmetic extends Arithmetic[SIntWrapper] {
    override implicit def cast(self: SIntWrapper): ArithmeticOps[SIntWrapper] = new ArithmeticOps(self) {
      override def name = s"s${self.bits.getWidth}"

      override def *(t: SIntWrapper) = SIntWrapper(self.bits.asSInt * t.bits.asSInt)
      override def /(t: SIntWrapper) = SIntWrapper(Mux(t.bits.asSInt === 0.S, 0.S, self.bits.asSInt / t.bits.asSInt))
      override def mac(m1: SIntWrapper, m2: SIntWrapper) = SIntWrapper(
        m1.bits.asSInt * m2.bits.asSInt + self.bits.asSInt
      )
      override def +(t: SIntWrapper) = SIntWrapper(self.bits.asSInt + t.bits.asSInt)
      override def -(t: SIntWrapper) = SIntWrapper(self.bits.asSInt - t.bits.asSInt)

      override def >>(u: UInt) = {
        // The equation we use can be found here: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm

        // TODO Do we need to explicitly handle the cases where "u" is a small number (like 0)? What is the default behavior here?
        val point_five = Mux(u === 0.U, 0.U, self.bits(u - 1.U))
        val zeros      = Mux(u <= 1.U, 0.U, self.asUInt & ((1.U << (u - 1.U)).asUInt - 1.U)) =/= 0.U
        val ones_digit = self.bits(u)

        val r = (point_five & (zeros | ones_digit)).asBool

        SIntWrapper((self.bits.asSInt >> u).asSInt + Mux(r, 1.S, 0.S))
      }

      override def >(t: SIntWrapper): Bool = self.bits.asSInt > t.bits.asSInt

      override def withWidthOf(t: SIntWrapper) = {
        if (self.getWidth >= t.getWidth)
          SIntWrapper(self.bits(t.getWidth - 1, 0).asSInt)
        else {
          val sign_bits = t.getWidth - self.getWidth
          val sign      = self.bits(self.getWidth - 1)
          val x         = Cat(Cat(Seq.fill(sign_bits)(sign)), self.bits).asUInt
          SIntWrapper(x)
        }
      }

      override def clippedToWidthOf(t: SIntWrapper): SIntWrapper = {
        val maxsat = ((1 << (t.getWidth - 1)) - 1).S
        val minsat = (-(1 << (t.getWidth - 1))).S
        val x = MuxCase(self, Seq((self.bits.asSInt > maxsat) -> maxsat, (self.bits.asSInt < minsat) -> minsat))
          .asUInt(t.getWidth - 1, 0)
          .asSInt
        SIntWrapper(x)
      }

      override def relu: SIntWrapper = SIntWrapper(Mux(self.bits.asSInt >= 0.S, self.bits.asSInt, 0.S))

      override def zero: SIntWrapper     = SIntWrapper(0.S(self.getWidth.W))
      override def identity: SIntWrapper = SIntWrapper(1.S(self.getWidth.W))
      override def minimum: SIntWrapper  = SIntWrapper((-(1 << (self.getWidth - 1))).S)
      override def abs: SIntWrapper      = SIntWrapper(Mux(self.bits.asSInt < 0.S, -self.bits.asSInt, self.bits.asSInt))
      override def neg: SIntWrapper      = SIntWrapper(-self.bits.asSInt)
    }
  }

  implicit object SIntArithmetic extends Arithmetic[SInt] {
    override implicit def cast(self: SInt): ArithmeticOps[SInt] = new ArithmeticOps(self) {
      override def name = s"s${self.getWidth}"

      override def *(t: SInt)              = self * t
      override def /(t: SInt)              = Mux(t === 0.S, 0.S.asTypeOf(self), self / t)
      override def mac(m1: SInt, m2: SInt) = m1 * m2 + self
      override def +(t: SInt)              = self + t
      override def -(t: SInt)              = self - t

      override def >>(u: UInt) = {
        // The equation we use can be found here: https://riscv.github.io/documents/riscv-v-spec/#_vector_fixed_point_rounding_mode_register_vxrm

        // TODO Do we need to explicitly handle the cases where "u" is a small number (like 0)? What is the default behavior here?
        val point_five = Mux(u === 0.U, 0.U, self(u - 1.U))
        val zeros      = Mux(u <= 1.U, 0.U, self.asUInt & ((1.U << (u - 1.U)).asUInt - 1.U)) =/= 0.U
        val ones_digit = self(u)

        val r = (point_five & (zeros | ones_digit)).asBool

        (self >> u).asSInt + Mux(r, 1.S, 0.S)
      }

      override def >(t: SInt): Bool = self > t

      override def withWidthOf(t: SInt) = {
        if (self.getWidth >= t.getWidth)
          self(t.getWidth - 1, 0).asSInt
        else {
          val sign_bits = t.getWidth - self.getWidth
          val sign      = self(self.getWidth - 1)
          Cat(Cat(Seq.fill(sign_bits)(sign)), self).asTypeOf(t)
        }
      }

      override def clippedToWidthOf(t: SInt): SInt = {
        val maxsat = ((1 << (t.getWidth - 1)) - 1).S
        val minsat = (-(1 << (t.getWidth - 1))).S
        MuxCase(self, Seq((self > maxsat) -> maxsat, (self < minsat) -> minsat))(t.getWidth - 1, 0).asSInt
      }

      override def relu: SInt = Mux(self >= 0.S, self, 0.S)

      override def zero: SInt     = 0.S
      override def identity: SInt = 1.S
      override def minimum: SInt  = (-(1 << (self.getWidth - 1))).S
      override def abs: SInt      = Mux(self < 0.S, -self, self)
      override def neg: SInt      = -self
    }
  }

  implicit object FloatArithmetic extends Arithmetic[Float] {
    // TODO Floating point arithmetic currently switches between recoded and standard formats for every operation. However, it should stay in the recoded format as it travels through the systolic array

    override implicit def cast(self: Float): ArithmeticOps[Float] = new ArithmeticOps(self) {
      private val minNativeSigWidth = 4

      private def computeType(values: Float*): Float =
        Float(values.map(_.expWidth).max, values.map(_.sigWidth).max.max(minNativeSigWidth))

      private def resizeTo(value: Float, target: Float): Float =
        if (value.expWidth == target.expWidth && value.sigWidth == target.sigWidth) value
        else value.clippedToWidthOf(target)

      private def recode(value: Float): UInt =
        recFNFromFN(value.expWidth, value.sigWidth, value.bits)

      private def fromRecoded(recoded: UInt, target: Float): Float = {
        val result = Wire(Float(target.expWidth, target.sigWidth))
        result.bits := fNFromRecFN(target.expWidth, target.sigWidth, recoded)
        result
      }

      private def narrowToSelf(value: Float): Float =
        if (value.expWidth == self.expWidth && value.sigWidth == self.sigWidth) value
        else value.clippedToWidthOf(self)

      override def name = {
        val total_width = self.expWidth + self.sigWidth
        if (total_width >= 16) {
          s"hffp${total_width}"
        } else {
          s"hffp${total_width}e${self.expWidth - 1}m${self.sigWidth}"
        }
      }

      override def *(t: Float): Float = {
        val opType = computeType(self, t)
        val lhs    = resizeTo(self, opType)
        val rhs    = resizeTo(t, opType)

        val muladder = Module(new MulRecFN(opType.expWidth, opType.sigWidth))

        muladder.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := recode(lhs)
        muladder.io.b := recode(rhs)

        narrowToSelf(fromRecoded(muladder.io.out, opType))
      }

      override def /(t: Float): Float = {
        val opType = computeType(self, t)
        val numer  = resizeTo(self, opType)
        val denom  = resizeTo(t, opType)
        val fnWidth = opType.expWidth + opType.sigWidth

        val div        = Module(new DivSqrtRecFN_small(opType.expWidth, opType.sigWidth, 0))
        val reqNum     = RegInit(0.U(fnWidth.W))
        val reqDenom   = RegInit(0.U(fnWidth.W))
        val reqPending = RegInit(false.B)

        val requestChanged = denom.bits =/= 0.U && (numer.bits =/= reqNum || denom.bits =/= reqDenom)
        val issueNum       = Mux(requestChanged, numer.bits, reqNum)
        val issueDenom     = Mux(requestChanged, denom.bits, reqDenom)
        val launch         = div.io.inReady && (reqPending || requestChanged)

        div.io.inValid        := launch
        div.io.sqrtOp         := false.B
        div.io.a              := recFNFromFN(opType.expWidth, opType.sigWidth, issueNum)
        div.io.b              := recFNFromFN(opType.expWidth, opType.sigWidth, issueDenom)
        div.io.roundingMode   := consts.round_near_even
        div.io.detectTininess := consts.tininess_afterRounding

        // ArithmeticOps exposes only a plain data result, so keep the most
        // recent completed quotient from the iterative HardFloat divider.
        val lastQuotRec = RegInit(0.U((opType.expWidth + opType.sigWidth + 1).W))
        when(requestChanged) {
          reqNum     := numer.bits
          reqDenom   := denom.bits
          reqPending := true.B
        }
        when(launch) {
          reqNum     := issueNum
          reqDenom   := issueDenom
          reqPending := false.B
        }
        when(div.io.outValid_div) {
          lastQuotRec := div.io.out
        }

        val out = Wire(Float(opType.expWidth, opType.sigWidth))
        out.bits := Mux(denom.bits === 0.U, 0.U, fNFromRecFN(opType.expWidth, opType.sigWidth, lastQuotRec))
        narrowToSelf(out)
      }

      override def mac(m1: Float, m2: Float): Float = {
        val opType = computeType(self, m1, m2)
        val lhs    = resizeTo(m1, opType)
        val rhs    = resizeTo(m2, opType)
        val acc    = resizeTo(self, opType)

        val muladder = Module(new MulAddRecFN(opType.expWidth, opType.sigWidth))

        muladder.io.op             := 0.U
        muladder.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := recode(lhs)
        muladder.io.b := recode(rhs)
        muladder.io.c := recode(acc)

        narrowToSelf(fromRecoded(muladder.io.out, opType))
      }

      override def +(t: Float): Float = {
        val opType = computeType(self, t)
        val lhs    = resizeTo(self, opType)
        val rhs    = resizeTo(t, opType)

        // Generate 1 as a float
        val in_to_rec_fn = Module(new INToRecFN(1, opType.expWidth, opType.sigWidth))
        in_to_rec_fn.io.signedIn       := false.B
        in_to_rec_fn.io.in             := 1.U
        in_to_rec_fn.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        in_to_rec_fn.io.detectTininess := consts.tininess_afterRounding

        val one_rec = in_to_rec_fn.io.out

        // Perform addition
        val muladder = Module(new MulAddRecFN(opType.expWidth, opType.sigWidth))

        muladder.io.op             := 0.U
        muladder.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := recode(rhs)
        muladder.io.b := one_rec
        muladder.io.c := recode(lhs)

        narrowToSelf(fromRecoded(muladder.io.out, opType))
      }

      override def -(t: Float): Float = {
        val t_sgn = t.bits(t.getWidth - 1)
        val neg_t = Cat(~t_sgn, t.bits(t.getWidth - 2, 0)).asTypeOf(t)
        self + neg_t
      }

      override def >>(u: UInt): Float = {
        val opType     = computeType(self)
        val lhs        = resizeTo(self, opType)
        val lhsRecoded = recode(lhs)

        // Get 2^(-u) as a recoded float
        val shift_exp = Wire(UInt(opType.expWidth.W))
        shift_exp := lhs.bias.U - u
        val shift_fn  = Cat(0.U(1.W), shift_exp, 0.U((opType.sigWidth - 1).W))
        val shift_rec = recFNFromFN(opType.expWidth, opType.sigWidth, shift_fn)

        assert(shift_exp =/= 0.U, "scaling by denormalized numbers is not currently supported")

        // Multiply self and 2^(-u)
        val muladder = Module(new MulRecFN(opType.expWidth, opType.sigWidth))

        muladder.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        muladder.io.detectTininess := consts.tininess_afterRounding

        muladder.io.a := lhsRecoded
        muladder.io.b := shift_rec

        narrowToSelf(fromRecoded(muladder.io.out, opType))
      }

      override def >(t: Float): Bool = {
        val opType = computeType(self, t)
        val lhs    = resizeTo(self, opType)
        val rhs    = resizeTo(t, opType)

        val comparator = Module(new CompareRecFN(opType.expWidth, opType.sigWidth))
        comparator.io.a         := recode(lhs)
        comparator.io.b         := recode(rhs)
        comparator.io.signaling := false.B

        comparator.io.gt
      }

      override def withWidthOf(t: Float): Float = {
        if (self.expWidth == t.expWidth) {
          if (self.sigWidth == t.sigWidth) {
            self
          } else if (self.sigWidth < t.sigWidth) {
            val result = Wire(Float(t.expWidth, t.sigWidth))
            result.bits := Cat(self.bits, 0.U((t.sigWidth - self.sigWidth).W))
            result
          } else {
            // If self's sigWidth is larger, we need to truncate it
            val result = Wire(Float(t.expWidth, t.sigWidth))
            result.bits := self.bits(self.getWidth - 1, self.getWidth - t.getWidth)
            result
          }
        } else {
          val self_rec = recFNFromFN(self.expWidth, self.sigWidth, self.bits)

          val resizer = Module(new RecFNToRecFN(self.expWidth, self.sigWidth, t.expWidth, t.sigWidth))
          resizer.io.in             := self_rec
          resizer.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
          resizer.io.detectTininess := consts.tininess_afterRounding

          val result = Wire(Float(t.expWidth, t.sigWidth))
          result.bits := fNFromRecFN(t.expWidth, t.sigWidth, resizer.io.out)
          result
        }
      }

      override def clippedToWidthOf(t: Float): Float = {
        // TODO check for overflow. Right now, we just assume that overflow doesn't happen
        val self_rec = recFNFromFN(self.expWidth, self.sigWidth, self.bits)

        val resizer = Module(new RecFNToRecFN(self.expWidth, self.sigWidth, t.expWidth, t.sigWidth))
        resizer.io.in             := self_rec
        resizer.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        resizer.io.detectTininess := consts.tininess_afterRounding

        val result = Wire(Float(t.expWidth, t.sigWidth))
        result.bits := fNFromRecFN(t.expWidth, t.sigWidth, resizer.io.out)
        result
      }

      override def relu: Float = {
        val raw = rawFloatFromFN(self.expWidth, self.sigWidth, self.bits)

        val result = Wire(Float(self.expWidth, self.sigWidth))
        result.bits := Mux(!raw.isZero && raw.sign, 0.U, self.bits)
        result
      }

      override def zero: Float = 0.U.asTypeOf(self)
      override def identity: Float =
        Cat(0.U(2.W), ~(0.U((self.expWidth - 1).W)), 0.U((self.sigWidth - 1).W)).asTypeOf(self)
      override def minimum: Float = Cat(1.U, ~(0.U(self.expWidth.W)), 0.U((self.sigWidth - 1).W)).asTypeOf(self)
      override def abs: Float = {
        Cat(0.U(1.W), self.bits(self.getWidth - 2, 0)).asTypeOf(self)
      }
      override def neg: Float = {
        val sgn = self.bits(self.getWidth - 1)
        Cat(~sgn, self.bits(self.getWidth - 2, 0)).asTypeOf(self)
      }

      override def splitForExp: Option[(SInt, UInt, Float)] = {
        // Split a non-positive fp number into its integer part and fractional part
        val sgn       = self.bits(self.getWidth - 1)
        val expRaw    = self.bits(self.getWidth - 2, self.getWidth - self.expWidth - 1)
        val expBias   = (1 << (self.expWidth - 1)) - 1
        val fracWidth = self.sigWidth - 1
        val frac      = self.bits(fracWidth - 1, 0)

        // Assume inputs are normal (non-subnormal) and non-positive as required by exp2.
        val isSubnormal = expRaw === 0.U
        val mant        = Cat(Mux(isSubnormal, 0.U(1.W), 1.U(1.W)), frac) // sigWidth bits
        val exp         = expRaw.zext - expBias.S

        val intWidth = self.expWidth + self.sigWidth
        val intMag   = Wire(UInt(intWidth.W))
        val fracMag  = Wire(UInt(fracWidth.W)) // magnitude of -x_f in Q0.(sigWidth-1)

        when(isSubnormal || mant === 0.U) {
          intMag  := 0.U
          fracMag := 0.U
        }.elsewhen(exp < 0.S) {
          intMag := 0.U
          val shift = (-exp).asUInt
          fracMag := (mant >> shift)(fracWidth - 1, 0)
        }.elsewhen(exp >= fracWidth.S) {
          val shift = (exp.asUInt - fracWidth.U)
          intMag  := (mant << shift)(intWidth - 1, 0)
          fracMag := 0.U
        }.otherwise {
          val shiftRight = (fracWidth.U - exp.asUInt)
          val intMagRaw  = mant >> shiftRight
          intMag := Cat(0.U((intWidth - self.sigWidth).W), intMagRaw)
          val mask    = (1.U << shiftRight) - 1.U
          val fracRaw = mant & mask
          fracMag := (fracRaw << exp.asUInt)(fracWidth - 1, 0)
        }

        val intMagS = Cat(0.U(1.W), intMag).asSInt
        val intPart = Mux(sgn, -intMagS, intMagS)

        // Build x_f as a Float: negative (or zero) fractional value in (-1, 0].
        val fracFp        = Wire(Float(self.expWidth, self.sigWidth))
        val fracZero      = fracMag === 0.U
        val lz            = countLeadingZeros(fracMag)
        val shift         = lz + 1.U
        val mantNorm      = (fracMag << shift)(fracWidth, 0)
        val expFrac       = (expBias.S - shift.zext).asUInt
        val fracExpField  = Mux(fracZero, 0.U, expFrac(self.expWidth - 1, 0))
        val fracFracField = Mux(fracZero, 0.U, mantNorm(self.sigWidth - 2, 0))
        val fracSign      = Mux(fracZero, 0.U, sgn)
        fracFp.bits := Cat(fracSign, fracExpField, fracFracField)

        Some((intPart, fracMag, fracFp))
      }

      override def isFp: Option[(Int, Int)] = Some((self.expWidth, self.sigWidth))

      override def divWithValid(t: Float, inValid: Bool): Option[ValidDivResult[Float]] = {
        val opType = computeType(self, t)
        val numer  = resizeTo(self, opType)
        val denom  = resizeTo(t, opType)
        val div    = Module(new DivSqrtRecFN_small(opType.expWidth, opType.sigWidth, 0))
        div.io.inValid        := inValid
        div.io.sqrtOp         := false.B
        div.io.a              := recode(numer)
        div.io.b              := recode(denom)
        div.io.roundingMode   := consts.round_near_even
        div.io.detectTininess := consts.tininess_afterRounding

        val accepted      = inValid && div.io.inReady
        val forceZeroResp = RegInit(false.B)
        when(accepted) {
          forceZeroResp := denom.bits === 0.U
        }.elsewhen(div.io.outValid_div) {
          forceZeroResp := false.B
        }

        val outBits = Wire(Float(opType.expWidth, opType.sigWidth))
        outBits.bits := Mux(
          forceZeroResp,
          0.U((opType.expWidth + opType.sigWidth).W),
          fNFromRecFN(opType.expWidth, opType.sigWidth, div.io.out)
        )

        val out = Wire(Valid(Float(self.expWidth, self.sigWidth)))
        out.valid := div.io.outValid_div
        out.bits  := narrowToSelf(outBits)

        Some(ValidDivResult(div.io.inReady, out))
      }
    }
  }

  implicit object DWFloatArithmetic extends Arithmetic[DWFloat] {
    // Use DesignWare FP IP

    override implicit def cast(self: DWFloat): ArithmeticOps[DWFloat] = new ArithmeticOps(self) {
      private val dwRoundNearestEven = 0.U(3.W)

      private def resizeToSelf(t: DWFloat): UInt = t.withWidthOf(self).bits

      override def name = {
        val total_width = self.expWidth + self.sigWidth
        if (total_width >= 16) {
          s"dwfp${total_width}"
        } else {
          s"dwfp${total_width}e${self.expWidth - 1}m${self.sigWidth}"
        }
      }

      override def *(t: DWFloat): DWFloat = {
        val mult = Module(new DW_fp_mult(self.sigWidth - 1, self.expWidth))
        mult.io.a   := self.bits
        mult.io.b   := resizeToSelf(t)
        mult.io.rnd := dwRoundNearestEven
        val result = Wire(DWFloat(self.expWidth, self.sigWidth))
        result.bits := mult.io.z
        result
      }

      override def /(t: DWFloat): DWFloat = {
        val denom = resizeToSelf(t)
        val div   = Module(new DW_fp_div(self.sigWidth - 1, self.expWidth))
        div.io.a   := self.bits
        div.io.b   := denom
        div.io.rnd := dwRoundNearestEven
        val result = Wire(DWFloat(self.expWidth, self.sigWidth))
        result.bits := Mux(denom === 0.U, 0.U, div.io.z)
        result
      }

      override def mac(m1: DWFloat, m2: DWFloat): DWFloat = {
        val mac = Module(new DW_fp_mac(self.sigWidth - 1, self.expWidth))
        mac.io.a   := resizeToSelf(m1)
        mac.io.b   := resizeToSelf(m2)
        mac.io.c   := self.bits
        mac.io.rnd := dwRoundNearestEven
        val result = Wire(DWFloat(self.expWidth, self.sigWidth))
        result.bits := mac.io.z
        result
      }

      override def +(t: DWFloat): DWFloat = {
        val adder = Module(new DW_fp_add(self.sigWidth - 1, self.expWidth))
        adder.io.a   := self.bits
        adder.io.b   := resizeToSelf(t)
        adder.io.rnd := dwRoundNearestEven
        val result = Wire(DWFloat(self.expWidth, self.sigWidth))
        result.bits := adder.io.z
        result
      }

      override def -(t: DWFloat): DWFloat = {
        val neg_t = Cat(~t.bits(self.getWidth - 1), t.bits(self.getWidth - 2, 0)).asTypeOf(t)
        self + neg_t
      }

      override def >>(u: UInt): DWFloat = {
        assert(u < self.bias.U, "scaling by denormalized numbers is not currently supported")

        val shiftExp = Wire(UInt(self.expWidth.W))
        shiftExp := self.bias.U - u

        val scale = Wire(DWFloat(self.expWidth, self.sigWidth))
        scale.bits := Cat(0.U(1.W), shiftExp, 0.U((self.sigWidth - 1).W))

        self * scale
      }

      override def >(t: DWFloat): Bool = {
        val cmp = Module(new DW_fp_cmp(self.sigWidth - 1, self.expWidth))
        cmp.io.a    := self.bits
        cmp.io.b    := resizeToSelf(t)
        cmp.io.zctr := false.B
        cmp.io.agtb
      }

      override def withWidthOf(t: DWFloat): DWFloat = {
        if (self.expWidth == t.expWidth) {
          if (self.sigWidth == t.sigWidth) {
            self
          } else if (self.sigWidth < t.sigWidth) {
            val result = Wire(DWFloat(t.expWidth, t.sigWidth))
            result.bits := Cat(self.bits, 0.U((t.sigWidth - self.sigWidth).W))
            result
          } else {
            // If self's sigWidth is larger, we need to truncate it
            val result = Wire(DWFloat(t.expWidth, t.sigWidth))
            result.bits := self.bits(self.getWidth - 1, self.getWidth - t.getWidth)
            result
          }
        } else {
          val self_rec = recFNFromFN(self.expWidth, self.sigWidth, self.bits)

          val resizer = Module(new RecFNToRecFN(self.expWidth, self.sigWidth, t.expWidth, t.sigWidth))
          resizer.io.in             := self_rec
          resizer.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
          resizer.io.detectTininess := consts.tininess_afterRounding

          val result = Wire(DWFloat(t.expWidth, t.sigWidth))
          result.bits := fNFromRecFN(t.expWidth, t.sigWidth, resizer.io.out)
          result
        }
      }

      override def clippedToWidthOf(t: DWFloat): DWFloat = {
        // TODO check for overflow. Right now, we just assume that overflow doesn't happen
        val self_rec = recFNFromFN(self.expWidth, self.sigWidth, self.bits)

        val resizer = Module(new RecFNToRecFN(self.expWidth, self.sigWidth, t.expWidth, t.sigWidth))
        resizer.io.in             := self_rec
        resizer.io.roundingMode   := consts.round_near_even // consts.round_near_maxMag
        resizer.io.detectTininess := consts.tininess_afterRounding

        val result = Wire(DWFloat(t.expWidth, t.sigWidth))
        result.bits := fNFromRecFN(t.expWidth, t.sigWidth, resizer.io.out)
        result
      }

      override def relu: DWFloat = {
        val raw    = rawFloatFromFN(self.expWidth, self.sigWidth, self.bits)
        val result = Wire(DWFloat(self.expWidth, self.sigWidth))
        result.bits := Mux(!raw.isZero && raw.sign, 0.U, self.bits)
        result
      }

      override def zero: DWFloat = 0.U.asTypeOf(self)
      override def identity: DWFloat =
        Cat(0.U(2.W), ~(0.U((self.expWidth - 1).W)), 0.U((self.sigWidth - 1).W)).asTypeOf(self)
      override def minimum: DWFloat = Cat(1.U, ~(0.U(self.expWidth.W)), 0.U((self.sigWidth - 1).W)).asTypeOf(self)
      override def abs: DWFloat = {
        Cat(0.U(1.W), self.bits(self.getWidth - 2, 0)).asTypeOf(self)
      }
      override def neg: DWFloat = {
        val sgn = self.bits(self.getWidth - 1)
        Cat(~sgn, self.bits(self.getWidth - 2, 0)).asTypeOf(self)
      }

      override def splitForExp: Option[(SInt, UInt, DWFloat)] = {
        val asFloat = Wire(Float(self.expWidth, self.sigWidth))
        asFloat.bits := self.bits

        Arithmetic.FloatArithmetic.cast(asFloat).splitForExp.map { case (intPart, fracMag, fracFp) =>
          val fracDw = Wire(DWFloat(self.expWidth, self.sigWidth))
          fracDw.bits := fracFp.bits
          (intPart, fracMag, fracDw)
        }
      }

      override def isFp: Option[(Int, Int)] = Some((self.expWidth, self.sigWidth))

      override def divWithValid(t: DWFloat, inValid: Bool): Option[ValidDivResult[DWFloat]] = {
        val denom = resizeToSelf(t)
        val div   = Module(new DW_fp_div(self.sigWidth - 1, self.expWidth))
        div.io.a   := self.bits
        div.io.b   := denom
        div.io.rnd := dwRoundNearestEven

        val accepted      = inValid
        val forceZeroResp = RegInit(false.B)
        when(accepted) {
          forceZeroResp := denom === 0.U
        }.elsewhen(RegNext(inValid, false.B)) {
          forceZeroResp := false.B
        }

        val resultReg = RegInit(0.U((self.expWidth + self.sigWidth).W))
        when(accepted) {
          resultReg := div.io.z
        }

        val outBits = Wire(DWFloat(self.expWidth, self.sigWidth))
        outBits.bits := Mux(forceZeroResp, 0.U((self.expWidth + self.sigWidth).W), resultReg)

        val out = Wire(Valid(DWFloat(self.expWidth, self.sigWidth)))
        out.valid := RegNext(inValid, false.B)
        out.bits  := outBits

        Some(ValidDivResult(true.B, out))
      }
    }
  }
}
