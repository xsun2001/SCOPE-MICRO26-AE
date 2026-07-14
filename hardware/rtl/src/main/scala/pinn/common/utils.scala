package pinn.common

import chisel3._
import chisel3.util._

class InputBuffer[T <: Data](size: Int, t: T) extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(DecoupledIO(Vec(size, t)))
    val out = Vec(size, DecoupledIO(t))
  })

  val bufs     = Seq.tabulate(size) { i => Module(new Queue(t, i + 1, pipe = true, flow = true)) }
  val allReady = bufs.map(_.io.enq.ready).reduce(_ && _)
  io.in.ready := allReady
  for (i <- 0 until size) {
    bufs(i).io.enq.bits  := io.in.bits(i)
    bufs(i).io.enq.valid := io.in.valid && allReady
    io.out(i)            <> bufs(i).io.deq
  }
}

object InputBuffer {
  def apply[T <: Data](t: T, in_vec: DecoupledIO[Vec[T]]): Seq[DecoupledIO[T]] = {
    val buf = Module(new InputBuffer(in_vec.bits.size, t))
    buf.io.in <> in_vec
    buf.io.out
  }
}

class OutputBuffer[T <: Data](size: Int, t: T) extends Module {
  val io = IO(new Bundle {
    val in  = Vec(size, Flipped(DecoupledIO(t)))
    val out = DecoupledIO(Vec(size, t))
  })

  val bufs     = Seq.tabulate(size) { i => Module(new Queue(t, size - i, pipe = true, flow = true)) }
  val allValid = bufs.map(_.io.deq.valid).reduce(_ && _)
  io.out.valid := allValid
  for (i <- 0 until size) {
    io.out.bits(i)       := bufs(i).io.deq.bits
    bufs(i).io.deq.ready := io.out.ready && allValid
    bufs(i).io.enq       <> io.in(i)
  }
}

object OutputBuffer {
  def apply[T <: Data](t: T, out_vec: DecoupledIO[Vec[T]]): Seq[DecoupledIO[T]] = {
    val buf = Module(new OutputBuffer(out_vec.bits.size, t))
    buf.io.out <> out_vec
    buf.io.in
  }
}

class SkewBuffer[T <: Data](size: Int, t: T, delayFn: Int => Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(size, t))
    val out = Output(Vec(size, t))
  })

  for (idx <- 0 until size) {
    io.out(idx) := ShiftRegister(io.in(idx), delayFn(idx))
  }
}

object SkewBuffer {
  def apply[T <: Data](in: Vec[T], delayFn: Int => Int): Vec[T] = {
    val buf = Module(new SkewBuffer(in.length, in(0).cloneType, delayFn))
    buf.io.in := in
    buf.io.out
  }
}

object Split {
  // Split 3
  def apply[T <: Data](t: T, width0: Int, width1: Int): (T, T, T) = {
    val width2 = t.getWidth - width0 - width1
    val x      = t.asUInt
    val c      = x(width2 - 1, 0)
    val b      = x(width1 + width2 - 1, width2)
    val a      = x(width0 + width1 + width2 - 1, width1 + width2)
    (a.asTypeOf(t), b.asTypeOf(t), c.asTypeOf(t))
  }
}
