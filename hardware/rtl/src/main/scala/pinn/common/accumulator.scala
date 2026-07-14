package pinn.common

import chisel3._
import chisel3.util._

object GenericAccumulatorOp extends ChiselEnum {
  val Hold, Load, Accumulate = Value
}

class ScratchpadControl(depth: Int) extends Bundle {
  val readAddr  = UInt(log2Ceil(depth).W)
  val readEn    = Bool()
  val writeAddr = UInt(log2Ceil(depth).W)
  val writeEn   = Bool()
}

class AccumulatorPEIO[T <: Data](accType: T) extends Bundle {
  val in                = Input(accType)
  val scratchpadStateIn = Input(accType)
  val useScratchpad     = Input(Bool())
  val op                = Input(GenericAccumulatorOp())
  val out               = Output(accType)
}

class AccumulatorPE[T <: Data: Arithmetic](accType: T) extends Module {
  override val desiredName = "AccumulatorPE"

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val io = IO(new AccumulatorPEIO(accType))

  val regState = RegInit(accType.zero)
  val base     = Mux(io.useScratchpad, io.scratchpadStateIn, regState)

  switch(io.op) {
    is(GenericAccumulatorOp.Load) {
      regState := Mux(io.useScratchpad, io.scratchpadStateIn, io.in)
    }
    is(GenericAccumulatorOp.Accumulate) {
      regState := base + io.in
    }
  }

  io.out := regState
}

class AccumulatorArrayIO[T <: Data](size: Int, accType: T) extends Bundle {
  val in                = Input(Vec(size, accType))
  val scratchpadStateIn = Input(Vec(size, accType))
  val useScratchpad     = Input(Bool())
  val op                = Input(GenericAccumulatorOp())
  val out               = Output(Vec(size, accType))
}

class AccumulatorArray[T <: Data: Arithmetic](size: Int, accType: T) extends Module {
  override val desiredName = "AccumulatorArray"

  val io = IO(new AccumulatorArrayIO(size, accType))

  val lanes = Seq.fill(size)(Module(new AccumulatorPE(accType)))
  for (i <- 0 until size) {
    lanes(i).io.in                := io.in(i)
    lanes(i).io.scratchpadStateIn := io.scratchpadStateIn(i)
    lanes(i).io.useScratchpad     := io.useScratchpad
    lanes(i).io.op                := io.op
    io.out(i)                     := lanes(i).io.out
  }
}
