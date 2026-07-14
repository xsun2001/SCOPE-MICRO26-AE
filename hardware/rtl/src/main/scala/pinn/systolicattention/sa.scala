package pinn.systolicattention

import chisel3._
import chisel3.util._
import pinn.common._

object PEOp extends ChiselEnum {
  val Nop, LoadL, LoadU, MacU, MacD, Exp2Pwl, MinU = Value
}

object CMPOp extends ChiselEnum {
  val Nop, Reset, SeedOldMax, RowmaxUpdate, StreamNewMax, StreamZero, StreamIntercept = Value
}

object ACCOp extends ChiselEnum {
  // AccSlMem is the second requested ACC_SL: sram_out <- sl * sram_in.
  val Nop, SetScale, ExpSa, AccSl, RecipSa, RecipSl, AccSa, AccSlMem = Value
}

private object SaPipe {
  def apply[T <: Data](in: T): T = RegNext(in)
}

object SystolicAttentionInputBufferMode extends ChiselEnum {
  val Skew, ReverseSkew, Pipeline = Value
}

class SystolicAttentionInputBuffer[T <: Data](lanes: Int, dataType: T) extends Module {
  require(lanes > 0, s"lanes must be positive, got $lanes")

  private val zeroLine = 0.U.asTypeOf(Vec(lanes, dataType.cloneType))

  val io = IO(new Bundle {
    val mode = Input(SystolicAttentionInputBufferMode())
    val in   = Input(Vec(lanes, dataType.cloneType))
    val out  = Output(Vec(lanes, dataType.cloneType))
  })

  private val skew = VecInit(Seq.tabulate(lanes) { lane =>
    ShiftRegister(io.in(lane), lane + 1)
  })
  private val reverseSkew = VecInit(Seq.tabulate(lanes) { lane =>
    ShiftRegister(io.in(lane), lanes - lane)
  })
  private val pipeline = RegNext(io.in, zeroLine)

  io.out := MuxLookup(io.mode.asUInt, skew)(
    Seq(
      SystolicAttentionInputBufferMode.Skew.asUInt        -> skew,
      SystolicAttentionInputBufferMode.ReverseSkew.asUInt -> reverseSkew,
      SystolicAttentionInputBufferMode.Pipeline.asUInt    -> pipeline
    )
  )
}

class PE[T <: Data: Arithmetic](dataType: T, pieces: Int = SystolicAttentionPwl.pieceCount) extends Module {
  val io = IO(new Bundle {
    val op      = Input(PEOp())
    val lInput  = Input(dataType.cloneType)
    val uInput  = Input(dataType.cloneType)
    val dInput  = Input(dataType.cloneType)
    val rOutput = Output(dataType.cloneType)
    val dOutput = Output(dataType.cloneType)
    val uOutput = Output(dataType.cloneType)
    val regOut  = Output(dataType.cloneType)
  })

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val zero    = dataType.zero
  private val regVal  = RegInit(zero)
  private val expUnit = Module(new SystolicAttentionScale(dataType, pieces))

  expUnit.io.cmd          := Mux(io.op === PEOp.Exp2Pwl, SystolicAttentionScaleCmd.Exp2, SystolicAttentionScaleCmd.Fma)
  expUnit.io.x            := Mux(io.op === PEOp.Exp2Pwl, regVal, io.lInput)
  expUnit.io.w            := Mux(io.op === PEOp.Exp2Pwl, io.lInput, regVal)
  expUnit.io.b            := Mux(io.op === PEOp.MacU, io.dInput, io.uInput)
  expUnit.io.coeffEncoded := io.op === PEOp.Exp2Pwl

  io.rOutput := io.lInput
  io.dOutput := io.uInput
  io.uOutput := io.dInput
  io.regOut  := regVal

  switch(io.op) {
    is(PEOp.LoadL) {
      regVal := io.lInput
    }
    is(PEOp.LoadU) {
      regVal     := io.uInput
      io.dOutput := io.uInput
    }
    is(PEOp.MacU) {
      io.uOutput := expUnit.io.out
    }
    is(PEOp.MacD) {
      io.dOutput := expUnit.io.out
    }
    is(PEOp.Exp2Pwl) {
      regVal     := expUnit.io.out
      io.dOutput := io.uInput
    }
    is(PEOp.MinU) {
      regVal     := regVal - io.uInput
      io.dOutput := io.uInput
    }
  }
}

class CMP[T <: Data: Arithmetic](dataType: T) extends Module {
  val io = IO(new Bundle {
    val op         = Input(CMPOp())
    val dInput     = Input(dataType.cloneType)
    val seedOldMax = Input(dataType.cloneType)
    val intercept  = Input(dataType.cloneType)
    val dOutput    = Output(dataType.cloneType)
    val oldMaxOut  = Output(dataType.cloneType)
    val newMaxOut  = Output(dataType.cloneType)
    val diffOut    = Output(dataType.cloneType)
  })

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val oldMax = RegInit(dataType.minimum)
  private val newMax = RegInit(dataType.minimum)

  switch(io.op) {
    is(CMPOp.Reset) {
      oldMax := dataType.minimum
      newMax := dataType.minimum
    }
    is(CMPOp.SeedOldMax) {
      oldMax := io.seedOldMax
      newMax := dataType.minimum
    }
    is(CMPOp.RowmaxUpdate) {
      newMax := Mux(io.dInput > newMax, io.dInput, newMax)
    }
  }

  io.dOutput := MuxLookup(io.op.asUInt, dataType.zero)(
    Seq(
      CMPOp.StreamNewMax.asUInt    -> newMax,
      CMPOp.StreamZero.asUInt      -> dataType.zero,
      CMPOp.StreamIntercept.asUInt -> io.intercept
    )
  )
  io.oldMaxOut := oldMax
  io.newMaxOut := newMax
  io.diffOut   := oldMax - newMax
}

class SystolicArray[T <: Data: Arithmetic](
    dataType: T,
    rows: Int,
    cols: Int,
    pieces: Int = SystolicAttentionPwl.pieceCount
) extends Module {
  val ev = implicitly[Arithmetic[T]]
  import ev._

  val io = IO(new Bundle {
    val peOp          = Input(Vec(rows, Vec(cols, PEOp())))
    val leftIn        = Input(Vec(rows, dataType.cloneType))
    val bottomIn      = Input(Vec(cols, dataType.cloneType))
    val cmpOp         = Input(Vec(cols, CMPOp()))
    val cmpSeedOldMax = Input(Vec(cols, dataType.cloneType))
    val cmpIntercept  = Input(Vec(cols, dataType.cloneType))
    val accOut        = Output(Vec(cols, dataType.cloneType))
    val rowMax        = Output(Vec(cols, dataType.cloneType))
    val rowDiff       = Output(Vec(cols, dataType.cloneType))
  })

  val cmpArray = Seq.fill(cols)(Module(new CMP(dataType)))
  val mesh     = Seq.fill(rows, cols)(Module(new PE(dataType, pieces)))
  val meshT    = mesh.transpose
  val zeroIn   = dataType.zero

  for (col <- 0 until cols) {
    cmpArray(col).io.op         := io.cmpOp(col)
    cmpArray(col).io.seedOldMax := io.cmpSeedOldMax(col)
    cmpArray(col).io.intercept  := io.cmpIntercept(col)
    io.rowMax(col)              := cmpArray(col).io.newMaxOut
    io.rowDiff(col)             := cmpArray(col).io.diffOut
  }

  for (row <- 0 until rows; col <- 0 until cols) {
    val pe  = mesh(row)(col)
    val lIn = if (col == 0) io.leftIn(row) else SaPipe(mesh(row)(col - 1).io.rOutput)
    val uIn = if (row == 0) cmpArray(col).io.dOutput else SaPipe(mesh(row - 1)(col).io.dOutput)
    val dIn = if (row == rows - 1) io.bottomIn(col) else SaPipe(mesh(row + 1)(col).io.uOutput)
    pe.io.op     := io.peOp(row)(col)
    pe.io.lInput := lIn
    pe.io.uInput := uIn
    pe.io.dInput := dIn
  }

  for (col <- 0 until cols) {
    val cmpIn = if (rows == 0) zeroIn else SaPipe(meshT(col).head.io.uOutput)
    cmpArray(col).io.dInput := cmpIn
    io.accOut(col)          := SaPipe(meshT(col).last.io.dOutput)
  }
}

class Accumulator[T <: Data: Arithmetic](
    accType: T,
    lanes: Int,
    depth: Int,
    pieces: Int = SystolicAttentionPwl.pieceCount
) extends Module {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(depth > 0, s"depth must be positive, got $depth")

  private val rowIdxW       = math.max(1, log2Ceil(depth))
  private val logicalAddrW  = log2Ceil(lanes * depth)
  private val physicalDepth = SramMacroConfig.accumulatorBankDepth(lanes).max(depth)
  private val zeroElem      = 0.U.asTypeOf(accType.cloneType)
  private val zeroLine      = 0.U.asTypeOf(Vec(lanes, accType.cloneType))
  private val oneLit        = SystolicAttentionPwl.oneBits(accType).U(accType.getWidth.W).asTypeOf(accType.cloneType)
  private def rowAddr(row: UInt): UInt = (row * lanes.U)(logicalAddrW - 1, 0)

  val io = IO(new Bundle {
    val op             = Input(Vec(lanes, ACCOp()))
    val alpha          = Input(Vec(lanes, accType.cloneType))
    val saIn           = Input(Vec(lanes, accType.cloneType))
    val sa             = Output(Vec(lanes, accType.cloneType))
    val sl             = Output(Vec(lanes, accType.cloneType))
    val saInRegOut     = Output(Vec(lanes, accType.cloneType))
    val sramOut        = Output(Vec(lanes, accType.cloneType))
    val sramReadRow    = Input(UInt(rowIdxW.W))
    val sramReadData   = Output(Vec(lanes, accType.cloneType))
    val sramWriteEn    = Input(Bool())
    val sramWriteRow   = Input(UInt(rowIdxW.W))
    val sramWriteData  = Input(Vec(lanes, accType.cloneType))
    val sramWriteUseOp = Input(Bool())
  })

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val saReg      = RegInit(zeroLine)
  val slReg      = RegInit(zeroLine)
  val saInReg    = RegInit(zeroLine)
  val sramReadReg = RegInit(zeroLine)
  val sramOutReg = RegInit(zeroLine)
  saInReg := io.saIn

  val sram = Module(
    new GroupedRowMem(1, SramMacroConfig.AccumulatorBankGroups, lanes, depth, physicalDepth, 1, accType.cloneType)
  )
  sram.io.rd(0).en   := true.B
  sram.io.rd(0).addr := rowAddr(io.sramReadRow)
  sram.io.wr.en      := io.sramWriteEn
  sram.io.wr.addr    := rowAddr(io.sramWriteRow)
  sram.io.wr.data    := Mux(io.sramWriteUseOp, sramOutReg, io.sramWriteData)
  sramReadReg        := sram.io.rd(0).data
  io.sramReadData    := sramReadReg

  val scaleUnits = Seq.fill(lanes)(Module(new SystolicAttentionScale(accType, pieces)))
  val expSlopeRegs =
    VecInit(SystolicAttentionPwl.rawSlopes(accType, pieces).map(_.U(accType.getWidth.W).asTypeOf(accType.cloneType)))
  val expInterceptRegs =
    VecInit(
      SystolicAttentionPwl.rawIntercepts(accType, pieces).map(_.U(accType.getWidth.W).asTypeOf(accType.cloneType))
    )
  val recipDestIsSl = RegInit(VecInit(Seq.fill(lanes)(false.B)))

  for (lane <- 0 until lanes) {
    val scaleCmd      = WireDefault(SystolicAttentionScaleCmd.Idle)
    val scaleX        = WireDefault(zeroElem)
    val scaleW        = WireDefault(zeroElem)
    val scaleB        = WireDefault(zeroElem)
    val slope         = Mux1H(UIntToOH(scaleUnits(lane).io.expIdx, pieces), expSlopeRegs)
    val intercept     = Mux1H(UIntToOH(scaleUnits(lane).io.expIdx, pieces), expInterceptRegs)
    val recipInIsSl   = io.op(lane) === ACCOp.RecipSl
    val recipInput    = Mux(recipInIsSl, slReg(lane), saReg(lane))
    val recip         = oneLit.divWithValid(recipInput, io.op(lane) === ACCOp.RecipSa || recipInIsSl).get
    val recipAccepted = recip.inReady && (io.op(lane) === ACCOp.RecipSa || recipInIsSl)

    switch(io.op(lane)) {
      is(ACCOp.ExpSa) {
        scaleCmd := SystolicAttentionScaleCmd.Exp2
        scaleX   := saReg(lane)
        scaleW   := slope
        scaleB   := intercept
      }
      is(ACCOp.AccSl) {
        scaleCmd := SystolicAttentionScaleCmd.Fma
        scaleX   := saReg(lane)
        scaleW   := slReg(lane)
        scaleB   := saInReg(lane)
      }
      is(ACCOp.AccSa) {
        scaleCmd := SystolicAttentionScaleCmd.Fma
        scaleX   := sramReadReg(lane)
        scaleW   := saReg(lane)
        scaleB   := saInReg(lane)
      }
      is(ACCOp.AccSlMem) {
        scaleCmd := SystolicAttentionScaleCmd.Mul
        scaleX   := sramReadReg(lane)
        scaleW   := slReg(lane)
      }
    }

    scaleUnits(lane).io.cmd          := scaleCmd
    scaleUnits(lane).io.x            := scaleX
    scaleUnits(lane).io.w            := scaleW
    scaleUnits(lane).io.b            := scaleB
    scaleUnits(lane).io.coeffEncoded := false.B

    when(recipAccepted) {
      recipDestIsSl(lane) := recipInIsSl
    }
    when(recip.out.valid) {
      when(recipDestIsSl(lane)) {
        slReg(lane) := recip.out.bits
      }.otherwise {
        saReg(lane) := recip.out.bits
      }
    }

    switch(io.op(lane)) {
      is(ACCOp.SetScale) {
        saReg(lane) := io.alpha(lane)
      }
      is(ACCOp.ExpSa) {
        saReg(lane) := scaleUnits(lane).io.out
      }
      is(ACCOp.AccSl) {
        slReg(lane) := scaleUnits(lane).io.out
      }
      is(ACCOp.AccSa) {
        sramOutReg(lane) := scaleUnits(lane).io.out
      }
      is(ACCOp.AccSlMem) {
        sramOutReg(lane) := scaleUnits(lane).io.out
      }
    }
  }

  io.sa         := saReg
  io.sl         := slReg
  io.saInRegOut := saInReg
  io.sramOut    := sramOutReg
}

class SystolicAttentionMemoryHostIO[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val rdAddr = Input(UInt(addrWidth.W))
  val rdData = Output(Vec(portWidth, dataType))
  val wrEn   = Input(Bool())
  val wrAddr = Input(UInt(addrWidth.W))
  val wrData = Input(Vec(portWidth, dataType))
}

object SystolicAttentionInputSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val External = 2.U(2.W)
}

object SystolicAttentionAccSaInSrc {
  val Zero = 0.U(2.W); val SaAccOut = 1.U(2.W); val External = 2.U(2.W)
}

class SystolicAttentionSystem[T <: Data: Arithmetic](
    N: Int,
    dataType: T,
    accType: T,
    scratchpadDepth: Int,
    accumulatorDepth: Int = 2,
    pieces: Int = SystolicAttentionPwl.pieceCount
) extends Module {
  require(N > 0 && scratchpadDepth >= N && accumulatorDepth > 0, "invalid SystolicAttentionSystem dimensions")

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val spAddrWidth = SramMacroConfig.scratchpadAddrWidth(N, scratchpadDepth)
  private val accRowIdxW  = math.max(1, log2Ceil(accumulatorDepth))
  private val zeroAccLine = 0.U.asTypeOf(Vec(N, accType.cloneType))

  val io = IO(new Bundle {
    val sp = new SystolicAttentionMemoryHostIO(spAddrWidth, N, dataType)
    val ctrl = Input(new Bundle {
      val spReadEn, spWriteEn, accWriteEn, accWriteUseOp, accAlphaExternal = Bool()
      val spReadAddr, spWriteAddr                                          = UInt(spAddrWidth.W)
      val spWriteData                                                      = Vec(N, dataType.cloneType)
      val leftSrc, bottomSrc, accSaInSrc                                   = UInt(2.W)
      val leftBufferMode, bottomBufferMode                                 = SystolicAttentionInputBufferMode()
      val leftExternal, bottomExternal                                     = Vec(N, accType.cloneType)
      val peOp                                                             = Vec(N, Vec(N, PEOp()))
      val cmpOp                                                            = Vec(N, CMPOp())
      val cmpSeedOldMax, cmpIntercept                                      = Vec(N, accType.cloneType)
      val accAlpha, accSaInExternal                                        = Vec(N, accType.cloneType)
      val accOp                                                            = Vec(N, ACCOp())
      val accReadRow, accWriteRow                                          = UInt(accRowIdxW.W)
      val accWriteData                                                     = Vec(N, accType.cloneType)
    })
  })

  val scratchpad    = Scratchpad(N, scratchpadDepth, 1, dataType)
  val systolicArray = Module(new SystolicArray(accType, N, N, pieces))
  val accum         = Module(new Accumulator(accType, N, accumulatorDepth, pieces))
  val leftBuffer    = Module(new SystolicAttentionInputBuffer(N, accType))
  val bottomBuffer  = Module(new SystolicAttentionInputBuffer(N, accType))
  val spCtrlReadAcc = VecInit(scratchpad.io.ctrlReadData(0).map(_.withWidthOf(accType)))
  def pickLine(sel: UInt, external: Vec[T]): Vec[T] =
    MuxLookup(sel, zeroAccLine)(
      Seq(SystolicAttentionInputSrc.Scratchpad -> spCtrlReadAcc, SystolicAttentionInputSrc.External -> external)
    )

  scratchpad.io.host.rd(0).addr := io.sp.rdAddr
  scratchpad.io.host.wr(0).en   := io.sp.wrEn
  scratchpad.io.host.wr(0).addr := io.sp.wrAddr
  scratchpad.io.host.wr(0).data := io.sp.wrData
  scratchpad.io.ctrlReadEn(0)   := io.ctrl.spReadEn
  scratchpad.io.ctrlReadAddr(0) := io.ctrl.spReadAddr
  scratchpad.io.ctrlWriteEn     := io.ctrl.spWriteEn
  scratchpad.io.ctrlWriteAddr   := io.ctrl.spWriteAddr
  scratchpad.io.ctrlWriteData   := io.ctrl.spWriteData
  io.sp.rdData                  := scratchpad.io.host.rd(0).data

  leftBuffer.io.mode   := io.ctrl.leftBufferMode
  leftBuffer.io.in     := pickLine(io.ctrl.leftSrc, io.ctrl.leftExternal)
  bottomBuffer.io.mode := io.ctrl.bottomBufferMode
  bottomBuffer.io.in   := pickLine(io.ctrl.bottomSrc, io.ctrl.bottomExternal)

  systolicArray.io.peOp          := io.ctrl.peOp
  systolicArray.io.leftIn        := leftBuffer.io.out
  systolicArray.io.bottomIn      := bottomBuffer.io.out
  systolicArray.io.cmpOp         := io.ctrl.cmpOp
  systolicArray.io.cmpSeedOldMax := io.ctrl.cmpSeedOldMax
  systolicArray.io.cmpIntercept  := io.ctrl.cmpIntercept

  accum.io.op    := io.ctrl.accOp
  accum.io.alpha := Mux(io.ctrl.accAlphaExternal, io.ctrl.accAlpha, systolicArray.io.rowDiff)
  accum.io.saIn := MuxLookup(io.ctrl.accSaInSrc, zeroAccLine)(
    Seq(
      SystolicAttentionAccSaInSrc.SaAccOut -> systolicArray.io.accOut,
      SystolicAttentionAccSaInSrc.External -> io.ctrl.accSaInExternal
    )
  )
  accum.io.sramReadRow    := io.ctrl.accReadRow
  accum.io.sramWriteEn    := io.ctrl.accWriteEn
  accum.io.sramWriteRow   := io.ctrl.accWriteRow
  accum.io.sramWriteData  := io.ctrl.accWriteData
  accum.io.sramWriteUseOp := io.ctrl.accWriteUseOp
}
