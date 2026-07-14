package pinn.common

import chisel3._

object TypedLiteral {
  private def integerToFpBits(value: BigInt, expWidth: Int, sigWidth: Int): BigInt = {
    val fracWidth = sigWidth - 1
    val totalWidth = expWidth + sigWidth
    val signBit = if (value < 0) BigInt(1) else BigInt(0)
    val absValue = value.abs

    if (absValue == 0) {
      0
    } else {
      val bias = (1 << (expWidth - 1)) - 1
      val maxExp = (1 << expWidth) - 1
      val exponent = absValue.bitLength - 1
      var biasedExp = exponent + bias

      val significand =
        if (exponent <= fracWidth) {
          absValue << (fracWidth - exponent)
        } else {
          val shift = exponent - fracWidth
          val base = absValue >> shift
          val rem = absValue & ((BigInt(1) << shift) - 1)
          val half = BigInt(1) << (shift - 1)
          val roundUp = rem > half || (rem == half && (base & 1) == 1)
          val rounded = if (roundUp) base + 1 else base

          if (rounded == (BigInt(1) << sigWidth)) {
            biasedExp += 1
            rounded >> 1
          } else {
            rounded
          }
        }

      if (biasedExp >= maxExp) {
        (signBit << (totalWidth - 1)) | (BigInt(maxExp) << fracWidth)
      } else {
        val fracMask = (BigInt(1) << fracWidth) - 1
        val frac = significand & fracMask
        (signBit << (totalWidth - 1)) | (BigInt(biasedExp) << fracWidth) | frac
      }
    }
  }

  def requireFuseMaxSupported[T <: Data](dataType: T): Unit =
    dataType match {
      case _: SInt | _: SIntWrapper | _: Float | _: DWFloat =>
      case _: UInt =>
        require(requirement = false, "FuseMax does not support UInt because its recurrence requires negative values")
      case other =>
        require(
          requirement = false,
          s"FuseMax only supports SInt, SIntWrapper, Float, and DWFloat; got ${other.getClass.getSimpleName}"
        )
    }

  def fromBigInt[T <: Data](value: BigInt, dataType: T): T = {
    requireFuseMaxSupported(dataType)

    dataType match {
      case t: SInt =>
        value.S(t.getWidth.W).asTypeOf(dataType)
      case t: SIntWrapper =>
        value.S(t.getWidth.W).asUInt.asTypeOf(dataType)
      case t: Float =>
        integerToFpBits(value, t.expWidth, t.sigWidth).U(t.getWidth.W).asTypeOf(dataType)
      case t: DWFloat =>
        integerToFpBits(value, t.expWidth, t.sigWidth).U(t.getWidth.W).asTypeOf(dataType)
      case _ =>
        throw new IllegalArgumentException("unreachable type guard in TypedLiteral.fromBigInt")
    }
  }

  def fromInt[T <: Data](value: Int, dataType: T): T = fromBigInt(BigInt(value), dataType)
}
