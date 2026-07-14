package pinn.common

import chisel3._
import chisel3.util._

class NNLut[T <: Data: Arithmetic](size: Int, dataType: T) extends Module {
  val ev = implicitly[Arithmetic[T]]
  import ev._

  val io = IO(new Bundle {
    val in        = Input(dataType)
    val out       = Output(dataType)
    val setup     = Input(Bool())
    val setupType = Input(UInt(2.W))
  })

  val bp = Reg(Vec(size, dataType))
  val s  = Reg(Vec(size, dataType))
  val t  = Reg(Vec(size, dataType))

  val setupIdx = RegInit(0.U(log2Ceil(size).W))

  when(io.setup) {
    when(io.setupType === 0.U) {
      bp(setupIdx) := io.in
      setupIdx     := setupIdx + 1.U
    }.elsewhen(io.setupType === 1.U) {
      s(setupIdx) := io.in
      setupIdx    := setupIdx + 1.U
    }.elsewhen(io.setupType === 2.U) {
      t(setupIdx) := io.in
      setupIdx    := setupIdx + 1.U
    }
  }

  when(setupIdx === (size - 1).U) {
    setupIdx := 0.U
  }

  val reg_x = RegNext(io.in)
  val idx   = PriorityEncoder(VecInit(bp.map(b => io.in > b)))
  val reg_s = RegNext(s(idx))
  val reg_t = RegNext(t(idx))
  val reg_o = RegNext(reg_t.mac(reg_x, reg_s))
  io.out := reg_o
}
