package pinn.fusemax

import chisel3._
import chisel3.util._
import pinn.common._

object FuseMaxRF2D {
  val BQK = 0
  val TMP = 1
  val SLN = 2
}

object FuseMaxRF1D {
  val RM_OLD  = 0
  val RM_NEW  = 1
  val PRM     = 2
  val TMP_X   = 3
  val RD_OLD  = 4
  val RD_NEW  = 5
  val RNV_OLD = 6
  val RNV_NEW = 7
  val TMP_MUL = 8
}

object FuseMaxPeSrc extends ChiselEnum {
  val Zero, Pipe, North, Scalar, Rf = Value
}

object FuseMaxPeOp extends ChiselEnum {
  val Nop, Mov, Add, Sub, Mul, Mac, Max, ReduceMax, ReduceSum, Div = Value
}

object FuseMaxWestSrc extends ChiselEnum {
  val Zero, ScratchpadQ, External = Value
}

object FuseMaxNorthSrc extends ChiselEnum {
  val Zero, ScratchpadBk, ScratchpadBv, External = Value
}

object FuseMaxScalarSrc extends ChiselEnum {
  val Zero, LaneSouth, External = Value
}

object FuseMaxSpWriteSrc extends ChiselEnum {
  val External, LaneDrain = Value
}

class FuseMaxPeCmd(rfEntries: Int) extends Bundle {
  private val idxWidth = math.max(1, log2Ceil(rfEntries))

  val en = Bool()

  val op = FuseMaxPeOp()

  val aSel = FuseMaxPeSrc()
  val aIdx = UInt(idxWidth.W)

  val bSel = FuseMaxPeSrc()
  val bIdx = UInt(idxWidth.W)

  val cSel = FuseMaxPeSrc()
  val cIdx = UInt(idxWidth.W)

  val dstIdx   = UInt(idxWidth.W)
  val writeDst = Bool()

  val southFromFu = Bool()
  val drainFromFu = Bool()
}

class FuseMax1DPEIO[T <: Data](rfEntries: Int, reduceFanIn: Int, accType: T) extends Bundle {
  val cmd = Input(new FuseMaxPeCmd(rfEntries))

  val in_w     = Input(accType)
  val reduceIn = Input(Vec(reduceFanIn, accType))

  val out_e = Output(accType)
  val out_s = Output(accType)
  val out_d = Output(accType)
}

class FuseMax1DPE[T <: Data: Arithmetic](accType: T, rfEntries: Int = 10, reduceFanIn: Int = 1) extends Module {
  val io = IO(new FuseMax1DPEIO(rfEntries, reduceFanIn, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val pipeReg        = RegInit(accType.zero)
  val reduceReg      = RegInit(accType.zero)
  val reduceOpReg    = RegInit(FuseMaxPeOp.Nop)
  val reduceValidReg = RegInit(false.B)
  val rf             = RegInit(VecInit(Seq.fill(rfEntries)(accType.zero)))

  pipeReg := io.in_w

  private def pick(sel: FuseMaxPeSrc.Type, idx: UInt): T = {
    val out = WireDefault(accType.zero)

    switch(sel) {
      is(FuseMaxPeSrc.Zero) { out := accType.zero }
      is(FuseMaxPeSrc.Pipe) { out := pipeReg }
      is(FuseMaxPeSrc.North) { out := accType.zero }
      is(FuseMaxPeSrc.Scalar) { out := accType.zero }
      is(FuseMaxPeSrc.Rf) { out := rf(idx) }
    }

    out
  }

  val a = pick(io.cmd.aSel, io.cmd.aIdx)
  val b = pick(io.cmd.bSel, io.cmd.bIdx)
  val c = pick(io.cmd.cSel, io.cmd.cIdx)

  private def vecMax(values: Seq[T]): T = values.reduceLeft((lhs, rhs) => Mux(rhs > lhs, rhs, lhs))
  private def vecSum(values: Seq[T]): T = values.reduceLeft(_ + _)

  val reduceSlice = WireDefault(accType.zero)
  switch(io.cmd.op) {
    is(FuseMaxPeOp.ReduceMax) {
      reduceSlice := vecMax((0 until reduceFanIn).map(idx => io.reduceIn(idx)))
    }
    is(FuseMaxPeOp.ReduceSum) {
      reduceSlice := vecSum((0 until reduceFanIn).map(idx => io.reduceIn(idx)))
    }
  }

  val reduceActive = io.cmd.en &&
    (io.cmd.op === FuseMaxPeOp.ReduceMax || io.cmd.op === FuseMaxPeOp.ReduceSum)
  val reduceContinue = reduceValidReg && reduceOpReg === io.cmd.op
  val reduceNext     = WireDefault(reduceSlice)

  when(io.cmd.op === FuseMaxPeOp.ReduceMax) {
    reduceNext := Mux(reduceContinue && reduceReg > reduceSlice, reduceReg, reduceSlice)
  }
  when(io.cmd.op === FuseMaxPeOp.ReduceSum) {
    reduceNext := Mux(reduceContinue, reduceReg + reduceSlice, reduceSlice)
  }

  val macIdentity    = accType.zero.identity
  val macNegIdentity = macIdentity.neg
  val sharedMacAcc   = WireDefault(accType.zero)
  val sharedMacM1    = WireDefault(accType.zero)
  val sharedMacM2    = WireDefault(macIdentity)

  switch(io.cmd.op) {
    is(FuseMaxPeOp.Add) {
      sharedMacAcc := a
      sharedMacM1  := macIdentity
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Sub) {
      sharedMacAcc := a
      sharedMacM1  := macNegIdentity
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Mul) {
      sharedMacAcc := accType.zero
      sharedMacM1  := a
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Mac) {
      sharedMacAcc := c
      sharedMacM1  := a
      sharedMacM2  := b
    }
  }
  val sharedMacOut = sharedMacAcc.mac(sharedMacM1, sharedMacM2)

  val fuOut = WireDefault(accType.zero)

  switch(io.cmd.op) {
    is(FuseMaxPeOp.Nop) { fuOut := accType.zero }
    is(FuseMaxPeOp.Mov) { fuOut := a }
    is(FuseMaxPeOp.Add) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Sub) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Mul) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Mac) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Max) { fuOut := Mux(b > a, b, a) }
    is(FuseMaxPeOp.ReduceMax) { fuOut := reduceNext }
    is(FuseMaxPeOp.ReduceSum) { fuOut := reduceNext }
    is(FuseMaxPeOp.Div) { fuOut := a / b }
  }

  when(reduceActive) {
    reduceReg      := reduceNext
    reduceOpReg    := io.cmd.op
    reduceValidReg := true.B
  }.elsewhen(io.cmd.en) {
    reduceOpReg    := FuseMaxPeOp.Nop
    reduceValidReg := false.B
  }

  when(io.cmd.en && io.cmd.writeDst) {
    rf(io.cmd.dstIdx) := fuOut
  }

  io.out_e := pipeReg
  io.out_s := Mux(io.cmd.en && io.cmd.southFromFu, fuOut, accType.zero)
  io.out_d := Mux(io.cmd.en && io.cmd.drainFromFu, fuOut, accType.zero)
}

class FuseMax2DPEIO[T <: Data](rfEntries: Int, dataType: T, accType: T) extends Bundle {
  val cmd = Input(new FuseMaxPeCmd(rfEntries))

  val in_w      = Input(dataType)
  val in_n      = Input(dataType)
  val in_scalar = Input(accType)

  val out_e = Output(dataType)
  val out_s = Output(dataType)
  val out_d = Output(accType)
}

class FuseMax2DPE[T <: Data: Arithmetic](dataType: T, accType: T, rfEntries: Int = 10) extends Module {
  val io = IO(new FuseMax2DPEIO(rfEntries, dataType, accType))

  private val ev = implicitly[Arithmetic[T]]
  import ev._

  private def dataToAcc(elem: T): T = elem.withWidthOf(accType)

  private val zeroData = 0.U.asTypeOf(dataType.cloneType)
  private val zeroAcc  = 0.U.asTypeOf(accType.cloneType)

  val pipeReg  = RegInit(zeroData)
  val northReg = RegInit(zeroData)
  val rf       = RegInit(VecInit(Seq.fill(rfEntries)(zeroAcc)))

  pipeReg  := io.in_w
  northReg := io.in_n

  private def pick(sel: FuseMaxPeSrc.Type, idx: UInt): T = {
    val out = WireDefault(zeroAcc)

    switch(sel) {
      is(FuseMaxPeSrc.Zero) { out := zeroAcc }
      is(FuseMaxPeSrc.Pipe) { out := dataToAcc(pipeReg) }
      is(FuseMaxPeSrc.North) { out := dataToAcc(northReg) }
      is(FuseMaxPeSrc.Scalar) { out := io.in_scalar }
      is(FuseMaxPeSrc.Rf) { out := rf(idx) }
    }

    out
  }

  val a = pick(io.cmd.aSel, io.cmd.aIdx)
  val b = pick(io.cmd.bSel, io.cmd.bIdx)
  val c = pick(io.cmd.cSel, io.cmd.cIdx)

  val macIdentity    = zeroAcc.identity
  val macNegIdentity = macIdentity.neg
  val sharedMacAcc   = WireDefault(zeroAcc)
  val sharedMacM1    = WireDefault(zeroAcc)
  val sharedMacM2    = WireDefault(macIdentity)

  switch(io.cmd.op) {
    is(FuseMaxPeOp.Add) {
      sharedMacAcc := a
      sharedMacM1  := macIdentity
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Sub) {
      sharedMacAcc := a
      sharedMacM1  := macNegIdentity
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Mul) {
      sharedMacAcc := zeroAcc
      sharedMacM1  := a
      sharedMacM2  := b
    }
    is(FuseMaxPeOp.Mac) {
      sharedMacAcc := c
      sharedMacM1  := a
      sharedMacM2  := b
    }
  }
  val sharedMacOut = sharedMacAcc.mac(sharedMacM1, sharedMacM2)

  val fuOut = WireDefault(zeroAcc)

  switch(io.cmd.op) {
    is(FuseMaxPeOp.Nop) { fuOut := zeroAcc }
    is(FuseMaxPeOp.Mov) { fuOut := a }
    is(FuseMaxPeOp.Add) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Sub) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Mul) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Mac) { fuOut := sharedMacOut }
    is(FuseMaxPeOp.Max) { fuOut := Mux(b > a, b, a) }
    is(FuseMaxPeOp.ReduceMax) { fuOut := zeroAcc }
    is(FuseMaxPeOp.ReduceSum) { fuOut := zeroAcc }
    is(FuseMaxPeOp.Div) { fuOut := zeroAcc }
  }

  when(
    io.cmd.en &&
      (io.cmd.op === FuseMaxPeOp.Div ||
        io.cmd.op === FuseMaxPeOp.ReduceMax ||
        io.cmd.op === FuseMaxPeOp.ReduceSum)
  ) {
    assert(false.B, "Division and streamed reduction are only supported in the FuseMax 1D PE")
  }

  when(io.cmd.en && io.cmd.writeDst) {
    rf(io.cmd.dstIdx) := fuOut
  }

  io.out_e := pipeReg
  io.out_s := Mux(io.cmd.en && io.cmd.southFromFu, fuOut.clippedToWidthOf(dataType), northReg)
  io.out_d := Mux(io.cmd.en && io.cmd.drainFromFu, fuOut, zeroAcc)
}

class FuseMaxBuffer[T <: Data](N: Int, dataType: T, delayFn: Int => Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(N, dataType))
    val out = Output(Vec(N, dataType))
  })

  for (idx <- 0 until N) {
    io.out(idx) := ShiftRegister(io.in(idx), delayFn(idx))
  }
}

object FuseMaxBuffer {
  def apply[T <: Data](in: Vec[T], delayFn: Int => Int): Vec[T] = {
    val buffer = Module(new FuseMaxBuffer(in.length, in(0).cloneType, delayFn))
    buffer.io.in := in
    buffer.io.out
  }
}

class FuseMax2DArrayIO[T <: Data](N: Int, dataType: T, accType: T, rfEntries: Int) extends Bundle {
  val cmd = Input(Vec(N, Vec(N, new FuseMaxPeCmd(rfEntries))))

  val west   = Input(Vec(N, dataType))
  val north  = Input(Vec(N, dataType))
  val scalar = Input(Vec(N, accType))

  val south = Output(Vec(N, dataType))
  val drain = Output(Vec(N, Vec(N, accType)))
}

class FuseMax2DArray[T <: Data: Arithmetic](N: Int, dataType: T, accType: T, rfEntries: Int = 10) extends Module {
  val io = IO(new FuseMax2DArrayIO(N, dataType, accType, rfEntries))

  val pe = Seq.fill(N, N)(Module(new FuseMax2DPE(dataType, accType, rfEntries)))

  private val bufferedWest  = FuseMaxBuffer(io.west, row => row + 1)
  private val bufferedNorth = FuseMaxBuffer(io.north, col => col + 1)
  private val bottomSouth   = Wire(Vec(N, dataType.cloneType))

  for {
    row <- 0 until N
    col <- 0 until N
  } {
    pe(row)(col).io.cmd       := io.cmd(row)(col)
    pe(row)(col).io.in_w      := (if (col == 0) bufferedWest(row) else pe(row)(col - 1).io.out_e)
    pe(row)(col).io.in_n      := (if (row == 0) bufferedNorth(col) else pe(row - 1)(col).io.out_s)
    pe(row)(col).io.in_scalar := io.scalar(row)

    io.drain(row)(col) := pe(row)(col).io.out_d
  }

  for (col <- 0 until N) {
    bottomSouth(col) := pe(N - 1)(col).io.out_s
  }

  io.south := FuseMaxBuffer(bottomSouth, col => N - col - 1)
}

class FuseMax1DArrayIO[T <: Data](N: Int, accType: T, rfEntries: Int) extends Bundle {
  val cmd = Input(Vec(N, new FuseMaxPeCmd(rfEntries)))

  val west     = Input(Vec(N, accType))
  val reduceIn = Input(Vec(N, Vec(N, accType)))

  val south = Output(Vec(N, accType))
  val drain = Output(Vec(N, accType))
}

class FuseMax1DArray[T <: Data: Arithmetic](N: Int, accType: T, rfEntries: Int = 10) extends Module {
  val io = IO(new FuseMax1DArrayIO(N, accType, rfEntries))

  val lanes = Seq.fill(N)(Module(new FuseMax1DPE(accType, rfEntries, reduceFanIn = N)))

  for (lane <- 0 until N) {
    lanes(lane).io.cmd      := io.cmd(lane)
    lanes(lane).io.in_w     := io.west(lane)
    lanes(lane).io.reduceIn := io.reduceIn(lane)

    io.south(lane) := lanes(lane).io.out_s
    io.drain(lane) := lanes(lane).io.out_d
  }
}

class FuseMaxMemoryHostIO[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val rdAddr = Input(UInt(addrWidth.W))
  val rdData = Output(Vec(portWidth, dataType))
  val wrEn   = Input(Bool())
  val wrAddr = Input(UInt(addrWidth.W))
  val wrData = Input(Vec(portWidth, dataType))
}

class FuseMaxSystemCtrl[T <: Data](addrWidth: Int, N: Int, dataType: T, accType: T, rfEntries: Int) extends Bundle {
  val spQReadEn       = Bool()
  val spQReadAddr     = UInt(addrWidth.W)
  val spNorthReadEn   = Bool()
  val spNorthReadAddr = UInt(addrWidth.W)
  val spWriteEn       = Bool()
  val spWriteAddr     = UInt(addrWidth.W)
  val spWriteSrc      = FuseMaxSpWriteSrc()
  val spWriteData     = Vec(N, dataType.cloneType)

  val westSrc      = FuseMaxWestSrc()
  val westExternal = Vec(N, dataType.cloneType)

  val northSrc      = FuseMaxNorthSrc()
  val northExternal = Vec(N, dataType.cloneType)

  val scalarSrc      = FuseMaxScalarSrc()
  val scalarExternal = Vec(N, accType.cloneType)

  val pe2dCmd        = Vec(N, Vec(N, new FuseMaxPeCmd(rfEntries)))
  val pe1dCmd        = Vec(N, new FuseMaxPeCmd(rfEntries))
  val laneInExternal = Vec(N, accType.cloneType)
}

class FuseMaxSystemIO[T <: Data](addrWidth: Int, N: Int, dataType: T, accType: T, rfEntries: Int) extends Bundle {
  val sp   = new FuseMaxMemoryHostIO(addrWidth, N, dataType)
  val ctrl = Input(new FuseMaxSystemCtrl(addrWidth, N, dataType, accType, rfEntries))

  val arraySouth = Output(Vec(N, dataType.cloneType))
  val laneOutS   = Output(Vec(N, accType.cloneType))
  val laneOutD   = Output(Vec(N, accType.cloneType))
}

class FuseMaxSystem[T <: Data: Arithmetic](
    N: Int,
    dataType: T,
    accType: T,
    rfEntries: Int = 10,
    scratchpadDepth: Int = 64
) extends Module {
  private val spAddrWidth = SramMacroConfig.scratchpadAddrWidth(N, scratchpadDepth)

  val io = IO(new FuseMaxSystemIO(spAddrWidth, N, dataType, accType, rfEntries))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val zeroDataLine = 0.U.asTypeOf(Vec(N, dataType.cloneType))
  private val zeroAccLine  = 0.U.asTypeOf(Vec(N, accType.cloneType))
  private val zeroAccMesh  = 0.U.asTypeOf(Vec(N, Vec(N, accType.cloneType)))

  private def accToData(elem: T): T = elem.clippedToWidthOf(dataType)

  val scratchpad = Scratchpad(N, scratchpadDepth, 2, dataType)
  val array2d    = Module(new FuseMax2DArray(N, dataType, accType, rfEntries))
  val array1d    = Module(new FuseMax1DArray(N, accType, rfEntries))

  assert(
    PopCount(Seq(io.ctrl.spQReadEn, io.ctrl.spNorthReadEn)) <= 1.U,
    "FuseMax scratchpad exposes one controller read path; q and north reads must be serialized"
  )

  val spQData             = scratchpad.io.ctrlReadData(0)
  val spNorthData         = scratchpad.io.ctrlReadData(1)
  val array2dDrainPipeReg = RegInit(zeroAccMesh)
  val laneDrainPipeReg    = RegInit(zeroAccLine)

  array2dDrainPipeReg := array2d.io.drain
  laneDrainPipeReg    := array1d.io.drain

  val laneDrainAsData = VecInit(laneDrainPipeReg.map(accToData))

  val scalarFeedbackReg = RegInit(zeroAccLine)
  scalarFeedbackReg := array1d.io.south

  val westData = WireDefault(zeroDataLine)
  switch(io.ctrl.westSrc) {
    is(FuseMaxWestSrc.ScratchpadQ) {
      westData := spQData
    }
    is(FuseMaxWestSrc.External) {
      westData := io.ctrl.westExternal
    }
  }

  val northData = WireDefault(zeroDataLine)
  switch(io.ctrl.northSrc) {
    is(FuseMaxNorthSrc.ScratchpadBk) {
      northData := spNorthData
    }
    is(FuseMaxNorthSrc.ScratchpadBv) {
      northData := spNorthData
    }
    is(FuseMaxNorthSrc.External) {
      northData := io.ctrl.northExternal
    }
  }

  val scalarData = WireDefault(zeroAccLine)
  switch(io.ctrl.scalarSrc) {
    is(FuseMaxScalarSrc.LaneSouth) {
      scalarData := scalarFeedbackReg
    }
    is(FuseMaxScalarSrc.External) {
      scalarData := io.ctrl.scalarExternal
    }
  }

  val spWriteData = WireDefault(io.ctrl.spWriteData)
  when(io.ctrl.spWriteSrc === FuseMaxSpWriteSrc.LaneDrain) {
    spWriteData := laneDrainAsData
  }

  scratchpad.io.host.rd(0).addr := io.sp.rdAddr
  scratchpad.io.host.wr(0).en   := io.sp.wrEn
  scratchpad.io.host.wr(0).addr := io.sp.wrAddr
  scratchpad.io.host.wr(0).data := io.sp.wrData
  scratchpad.io.ctrlReadEn(0)   := io.ctrl.spQReadEn
  scratchpad.io.ctrlReadAddr(0) := io.ctrl.spQReadAddr
  scratchpad.io.ctrlReadEn(1)   := io.ctrl.spNorthReadEn
  scratchpad.io.ctrlReadAddr(1) := io.ctrl.spNorthReadAddr
  scratchpad.io.ctrlWriteEn     := io.ctrl.spWriteEn
  scratchpad.io.ctrlWriteAddr   := io.ctrl.spWriteAddr
  scratchpad.io.ctrlWriteData   := spWriteData
  io.sp.rdData                  := scratchpad.io.host.rd(0).data

  array2d.io.cmd    := io.ctrl.pe2dCmd
  array2d.io.west   := westData
  array2d.io.north  := northData
  array2d.io.scalar := scalarData

  array1d.io.cmd      := io.ctrl.pe1dCmd
  array1d.io.west     := io.ctrl.laneInExternal
  array1d.io.reduceIn := array2dDrainPipeReg

  io.arraySouth := array2d.io.south
  io.laneOutS   := array1d.io.south
  io.laneOutD   := laneDrainPipeReg
}
