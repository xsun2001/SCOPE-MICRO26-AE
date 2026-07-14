package pinn.common

import chisel3._
import chisel3.util._

class DW_fp_cmp(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 0
      )
    ) {
  val io = FlatIO(new Bundle {
    val a         = Input(UInt((sig_width + exp_width + 1).W))
    val b         = Input(UInt((sig_width + exp_width + 1).W))
    val zctr      = Input(Bool())
    val aeqb      = Output(Bool())
    val altb      = Output(Bool())
    val agtb      = Output(Bool())
    val unordered = Output(Bool())
    val z0        = Output(UInt((sig_width + exp_width + 1).W))
    val z1        = Output(UInt((sig_width + exp_width + 1).W))
    val status0   = Output(UInt(8.W))
    val status1   = Output(UInt(8.W))
  })
}

class DW_fp_add(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 0
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val b      = Input(UInt((sig_width + exp_width + 1).W))
    val rnd    = Input(UInt(3.W)) // Rounding mode
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}

class DW_fp_mult(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 0,
        "en_ubr_flag"     -> 0
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val b      = Input(UInt((sig_width + exp_width + 1).W))
    val rnd    = Input(UInt(3.W)) // Rounding mode
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}

class DW_fp_mac(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 0
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val b      = Input(UInt((sig_width + exp_width + 1).W))
    val c      = Input(UInt((sig_width + exp_width + 1).W))
    val rnd    = Input(UInt(3.W)) // Rounding mode
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}

class DW_fp_div(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 0,
        "faithful_round"  -> 1
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val b      = Input(UInt((sig_width + exp_width + 1).W))
    val rnd    = Input(UInt(3.W))
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}

class DW_fp_exp(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 1,
        "arch"            -> 1
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}

class DW_fp_exp2(val sig_width: Int, val exp_width: Int)
    extends ExtModule(
      Map(
        "sig_width"       -> sig_width,
        "exp_width"       -> exp_width,
        "ieee_compliance" -> 1,
        "arch"            -> 1
      )
    ) {
  val io = FlatIO(new Bundle {
    val a      = Input(UInt((sig_width + exp_width + 1).W))
    val z      = Output(UInt((sig_width + exp_width + 1).W))
    val status = Output(UInt(8.W))
  })
}
