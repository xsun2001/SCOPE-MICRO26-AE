package pinn.pinnacle

import chisel3._
import chisel3.util._
import pinn.common._

object PinnacleConfig {
  val StripCount = 2

  def supports(N: Int, stripHeight: Int): Boolean =
    N % 2 == 0 && stripHeight > 0 && StripCount * stripHeight <= N

  def requireSupportedGeometry(N: Int, stripHeight: Int): Unit = {
    require(N % 2 == 0, s"Pinnacle requires even N, got $N")
    require(stripHeight > 0, s"Pinnacle requires positive stripHeight, got $stripHeight")
    require(StripCount * stripHeight <= N, s"Pinnacle requires two strips: strip rows (${StripCount * stripHeight}) exceed N=$N")
  }

  final case class Geometry(N: Int, stripHeight: Int) {
    requireSupportedGeometry(N, stripHeight)

    val pairCount: Int   = N / 2
    val stripRows: Int   = StripCount * stripHeight
    val pinnLatency: Int = stripHeight + 1
  }
}

object WestOp extends ChiselEnum {
  val Hold, Preload, Compute, PinnCoeff = Value
}

object NorthOp extends ChiselEnum {
  val Hold, Pinn = Value
}

object AccOp extends ChiselEnum {
  val Nop, Reset, LoadSa, LoadSp, StoreSa, StoreSaReduceMax, StoreSaReduceSum, LoadReduce, LoadScale, StoreReduce,
      StoreScale, ClearReduceZero, ClearReduceMin, ClearScaleZero, ClearScaleMin, SubRowReduceToScale, StoreSaScale =
    Value
}

private object RawBitsCast {
  def apply[T <: Data, U <: Data](in: T, proto: U): U = {
    val out    = Wire(proto.cloneType)
    val inBits = in.asUInt
    if (in.getWidth >= proto.getWidth) {
      out := inBits(proto.getWidth - 1, 0).asTypeOf(proto.cloneType)
    } else {
      out := Cat(0.U((proto.getWidth - in.getWidth).W), inBits).asTypeOf(proto.cloneType)
    }
    out
  }
}

class WestToken[T <: Data](dataType: T) extends Bundle {
  val op   = WestOp()
  val data = dataType.cloneType
}

class NorthToken[T <: Data](accType: T) extends Bundle {
  val op   = NorthOp()
  val data = accType.cloneType
}

class PEStandardIO[T <: Data](dataType: T, accType: T) extends Bundle {
  val westOpIn = Input(WestOp())
  val westIn   = Input(dataType)
  val northIn  = Input(accType)
  val eastOut  = Output(dataType)
  val southOut = Output(accType)
}

class PEStandard[T <: Data: Arithmetic](dataType: T, accType: T) extends Module {
  val io = IO(new PEStandardIO(dataType, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val regE = RegInit(0.U.asTypeOf(dataType.cloneType))
  val regS = RegInit(0.U.asTypeOf(accType.cloneType))
  val regW = RegInit(0.U.asTypeOf(dataType.cloneType))

  val macOut = io.northIn.mac(regW.withWidthOf(accType), io.westIn.withWidthOf(accType))

  io.eastOut  := Mux(io.westOpIn === WestOp.Preload, regW, regE)
  io.southOut := regS

  switch(io.westOpIn) {
    is(WestOp.Preload) {
      regW := io.westIn
    }
    is(WestOp.Compute) {
      regE := io.westIn
      regS := macOut
    }
    is(WestOp.PinnCoeff) {
      regE := io.westIn
    }
  }
}

class PELeftIO[T <: Data](dataType: T, accType: T) extends Bundle {
  val westOpIn  = Input(WestOp())
  val westIn    = Input(dataType)
  val northOpIn = Input(NorthOp())
  val northIn   = Input(accType)
  val inRightB  = Input(dataType)
  val eastOut   = Output(dataType)
  val southOut  = Output(accType)
  val outRightP = Output(accType)
}

class PELeft[T <: Data: Arithmetic](dataType: T, accType: T) extends Module {
  val io = IO(new PELeftIO(dataType, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val regE = RegInit(0.U.asTypeOf(dataType.cloneType))
  val regS = RegInit(0.U.asTypeOf(accType.cloneType))
  val regW = RegInit(0.U.asTypeOf(dataType.cloneType))
  val regX = RegInit(0.U.asTypeOf(dataType.cloneType))

  val computeActive = io.westOpIn === WestOp.Compute
  val pinnActive    = io.northOpIn === NorthOp.Pinn
  val stripX        = RawBitsCast(io.northIn, dataType.cloneType)
  val forwardX      = RawBitsCast(regX, accType.cloneType)

  val sharedMacAcc = WireDefault(io.northIn)
  val sharedMacM1  = WireDefault(regW.withWidthOf(accType))
  val sharedMacM2  = WireDefault(io.westIn.withWidthOf(accType))

  when(pinnActive) {
    sharedMacAcc := io.inRightB.withWidthOf(accType)
    sharedMacM1  := stripX.withWidthOf(accType)
    sharedMacM2  := regE.withWidthOf(accType)
  }
  val sharedMacOut = sharedMacAcc.mac(sharedMacM1, sharedMacM2)

  assert(
    !(pinnActive && (io.westOpIn === WestOp.Compute || io.westOpIn === WestOp.PinnCoeff)),
    "PELeft reuses one MAC and regE; pinn compute only overlaps with preload or hold"
  )

  io.eastOut   := Mux(io.westOpIn === WestOp.Preload, regW, regE)
  io.southOut  := Mux(pinnActive, forwardX, regS)
  io.outRightP := regS

  switch(io.westOpIn) {
    is(WestOp.Preload) {
      regW := io.westIn
    }
    is(WestOp.Compute) {
      regE := io.westIn
    }
    is(WestOp.PinnCoeff) {
      regE := io.westIn
    }
  }

  when(computeActive || pinnActive) {
    regS := sharedMacOut
  }
  when(pinnActive) {
    regX := stripX
  }
}

class PERightIO[T <: Data](dataType: T, accType: T) extends Bundle {
  val westOpIn  = Input(WestOp())
  val westIn    = Input(dataType)
  val northOpIn = Input(NorthOp())
  val northIn   = Input(accType)
  val inLeftP   = Input(accType)
  val eastOut   = Output(dataType)
  val southOut  = Output(accType)
  val outLeftB  = Output(dataType)
}

class PERight[T <: Data: Arithmetic](dataType: T, accType: T) extends Module {
  val io = IO(new PERightIO(dataType, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val regE = RegInit(0.U.asTypeOf(dataType.cloneType))
  val regS = RegInit(0.U.asTypeOf(accType.cloneType))
  val regW = RegInit(0.U.asTypeOf(dataType.cloneType))

  val computeActive = io.westOpIn === WestOp.Compute
  val pinnActive    = io.northOpIn === NorthOp.Pinn
  val reluP         = io.inLeftP.relu

  val sharedMacAcc = WireDefault(io.northIn)
  val sharedMacM1  = WireDefault(regW.withWidthOf(accType))
  val sharedMacM2  = WireDefault(io.westIn.withWidthOf(accType))

  when(pinnActive) {
    sharedMacM1 := regS.identity
    sharedMacM2 := reluP
  }
  val sharedMacOut = sharedMacAcc.mac(sharedMacM1, sharedMacM2)

  assert(
    !(pinnActive && (io.westOpIn === WestOp.Compute || io.westOpIn === WestOp.PinnCoeff)),
    "PERight reuses one MAC and regE; pinn compute only overlaps with preload or hold"
  )

  io.eastOut  := Mux(io.westOpIn === WestOp.Preload, regW, regE)
  io.southOut := regS
  io.outLeftB := regE

  switch(io.westOpIn) {
    is(WestOp.Preload) {
      regW := io.westIn
    }
    is(WestOp.Compute) {
      regE := io.westIn
    }
    is(WestOp.PinnCoeff) {
      regE := io.westIn
    }
  }

  when(computeActive || pinnActive) {
    regS := sharedMacOut
  }
}

class StripPairIO[T <: Data](dataType: T, accType: T) extends Bundle {
  val westOpIn  = Input(WestOp())
  val westIn    = Input(dataType)
  val northOpIn = Input(NorthOp())
  val northIn   = Input(Vec(2, accType))
  val eastOut   = Output(dataType)
  val southOut  = Output(Vec(2, accType))
}

class StripPair[T <: Data: Arithmetic](dataType: T, accType: T) extends Module {
  val io = IO(new StripPairIO(dataType, accType))

  val left  = Module(new PELeft(dataType, accType))
  val right = Module(new PERight(dataType, accType))

  left.io.westOpIn  := io.westOpIn
  left.io.westIn    := io.westIn
  left.io.northOpIn := io.northOpIn
  left.io.northIn   := io.northIn(0)
  left.io.inRightB  := right.io.outLeftB

  right.io.westOpIn  := io.westOpIn
  right.io.westIn    := left.io.eastOut
  right.io.northOpIn := io.northOpIn
  right.io.northIn   := io.northIn(1)
  right.io.inLeftP   := left.io.outRightP

  io.eastOut  := right.io.eastOut
  io.southOut := VecInit(left.io.southOut, right.io.southOut)
}

class Buffer[T <: Data](N: Int, dataType: T, delayFn: Int => Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(N, dataType))
    val out = Output(Vec(N, dataType))
  })

  for (idx <- 0 until N) {
    io.out(idx) := ShiftRegister(io.in(idx), delayFn(idx))
  }
}

object Buffer {
  def apply[T <: Data](in: Vec[T], delayFn: Int => Int): Vec[T] = {
    val buffer = Module(new Buffer(in.length, in(0).cloneType, delayFn))
    buffer.io.in := in
    buffer.io.out
  }
}

class SystolicArrayIO[T <: Data](N: Int, dataType: T, accType: T) extends Bundle {
  val west      = Input(Vec(N, new WestToken(dataType)))
  val north     = Input(Vec(N, new NorthToken(accType)))
  val gemmSouth = Output(Vec(N, accType))
  val pinnSouth = Output(Vec(N, accType))
  val pinnValid = Output(Vec(N, Bool()))
}

class SystolicArray[T <: Data: Arithmetic](N: Int, dataType: T, accType: T, stripHeight: Int) extends Module {
  private val geometry = PinnacleConfig.Geometry(N, stripHeight)

  val io = IO(new SystolicArrayIO(N, dataType, accType))

  private val zeroAcc = 0.U.asTypeOf(accType.cloneType)

  private def isStripRow(row: Int): Boolean            = row < geometry.stripRows
  private def isStripTopRow(row: Int): Boolean         = isStripRow(row) && row % stripHeight == 0
  private def stripLane(stripIdx: Int, pair: Int): Int = stripIdx * geometry.pairCount + pair
  private def stripLatency(stripIdx: Int): Int         = geometry.pinnLatency + stripIdx * stripHeight
  private def deskewDelay(col: Int): Int               = N - col - 1

  val bufferedWest      = Buffer(io.west, row => row + 1)
  val stripSouthRegs    = Wire(Vec(geometry.stripRows, Vec(geometry.pairCount, Vec(2, accType.cloneType))))
  val gemmSouthRegs     = Wire(Vec(N, Vec(N, accType.cloneType)))
  val bottomGemmSouth   = Wire(Vec(N, accType.cloneType))
  val deskewedGemmSouth = Buffer(bottomGemmSouth, deskewDelay)

  for (row <- 0 until N) {
    if (isStripRow(row)) {
      val pairs = Seq.fill(geometry.pairCount)(Module(new StripPair(dataType, accType)))
      for (pair <- 0 until geometry.pairCount) {
        val lane0      = pair * 2
        val lane1      = lane0 + 1
        val stripIdx   = row / stripHeight
        val northLane  = stripLane(stripIdx, pair)
        val westIn     = if (pair == 0) bufferedWest(row).data else pairs(pair - 1).io.eastOut
        val northOpIn  = ShiftRegister(io.north(northLane).op, row + 1)
        val topNorthIn = ShiftRegister(io.north(northLane).data, row + 1)
        val pairNorthIn = WireDefault(
          VecInit(
            Seq(
              if (row == 0) zeroAcc else gemmSouthRegs(row - 1)(lane0),
              if (row == 0) zeroAcc else gemmSouthRegs(row - 1)(lane1)
            )
          )
        )

        if (isStripTopRow(row)) {
          when(northOpIn === NorthOp.Pinn) {
            pairNorthIn(0) := topNorthIn
            pairNorthIn(1) := zeroAcc
          }
        } else {
          when(northOpIn === NorthOp.Pinn) {
            pairNorthIn := stripSouthRegs(row - 1)(pair)
          }
        }

        pairs(pair).io.westOpIn  := bufferedWest(row).op
        pairs(pair).io.westIn    := westIn
        pairs(pair).io.northOpIn := northOpIn
        pairs(pair).io.northIn   := pairNorthIn

        gemmSouthRegs(row)(lane0) := pairs(pair).io.southOut(0)
        gemmSouthRegs(row)(lane1) := pairs(pair).io.southOut(1)
        stripSouthRegs(row)(pair) := pairs(pair).io.southOut
      }
    } else {
      val pes = Seq.fill(N)(Module(new PEStandard(dataType, accType)))
      for (col <- 0 until N) {
        val westIn  = if (col == 0) bufferedWest(row).data else pes(col - 1).io.eastOut
        val northIn = if (row == 0) zeroAcc else gemmSouthRegs(row - 1)(col)

        pes(col).io.westOpIn := bufferedWest(row).op
        pes(col).io.westIn   := westIn
        pes(col).io.northIn  := northIn

        gemmSouthRegs(row)(col) := pes(col).io.southOut
      }
    }
  }

  for (col <- 0 until N) {
    bottomGemmSouth(col) := gemmSouthRegs(N - 1)(col)
  }

  io.gemmSouth := deskewedGemmSouth
  io.pinnSouth := VecInit(Seq.fill(N)(zeroAcc))
  io.pinnValid := VecInit(Seq.fill(N)(false.B))

  for (stripIdx <- 0 until PinnacleConfig.StripCount) {
    val bottomRow = (stripIdx + 1) * stripHeight - 1
    for (pair <- 0 until geometry.pairCount) {
      val lane = stripLane(stripIdx, pair)
      io.pinnSouth(lane) := stripSouthRegs(bottomRow)(pair)(1)
      io.pinnValid(lane) := ShiftRegister(io.north(lane).op === NorthOp.Pinn, stripLatency(stripIdx) + 1)
    }
  }
}

class AccumulatorIO[T <: Data](lanes: Int, depth: Int, accType: T) extends Bundle {
  private val rowIdxW = math.max(1, log2Ceil(depth))

  val op            = Input(AccOp())
  val saIn          = Input(Vec(lanes, accType))
  val streamReadRow = Input(UInt(rowIdxW.W))
  val opReadRow     = Input(UInt(rowIdxW.W))
  val writeRow      = Input(UInt(rowIdxW.W))
  val saOut         = Output(Vec(lanes, accType))
  val spOut         = Output(Vec(lanes, accType))
}

class Accumulator[T <: Data: Arithmetic](accType: T, lanes: Int, depth: Int) extends Module {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(depth > 0, s"depth must be positive, got $depth")

  val io = IO(new AccumulatorIO(lanes, depth, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val zeroElem = 0.U.asTypeOf(accType.cloneType)
  private val zeroLine = 0.U.asTypeOf(Vec(lanes, accType.cloneType))
  private val minLine  = VecInit(Seq.fill(lanes)(accType.minimum))

  val banks = Seq.fill(lanes)(SyncReadMem(depth, accType.cloneType))

  val reduceReg  = RegInit(zeroLine)
  val scaleReg   = RegInit(zeroLine)
  val sramOutReg = RegInit(zeroLine)

  val streamIssue = io.op === AccOp.LoadSa || io.op === AccOp.LoadSp
  val workReadIssue = io.op === AccOp.LoadReduce || io.op === AccOp.LoadScale ||
    io.op === AccOp.SubRowReduceToScale || io.op === AccOp.StoreSaScale
  val workWriteIssue = io.op === AccOp.StoreSa || io.op === AccOp.StoreSaReduceMax ||
    io.op === AccOp.StoreSaReduceSum || io.op === AccOp.StoreReduce || io.op === AccOp.StoreScale

  val readIssue = streamIssue || workReadIssue
  val readAddr  = Mux(streamIssue, io.streamReadRow, io.opReadRow)
  val readData  = Wire(Vec(lanes, accType.cloneType))

  assert(!(streamIssue && workReadIssue), "Accumulator exposes one SRAM read port; stream and work reads must be serialized")

  for (lane <- 0 until lanes) {
    readData(lane) := banks(lane).read(readAddr, readIssue)
  }

  val streamCaptureEn = RegNext(streamIssue, false.B)
  when(streamCaptureEn) {
    sramOutReg := readData
  }

  val workReadPending = RegNext(workReadIssue, false.B)
  val workPendingOp   = RegEnable(io.op, workReadIssue)
  val workRowPipe     = RegEnable(io.writeRow, workReadIssue)
  val saInPipe        = RegEnable(io.saIn, io.op === AccOp.StoreSaScale)
  val macSel          = WireDefault(io.op)
  when(workReadPending && (workPendingOp === AccOp.SubRowReduceToScale || workPendingOp === AccOp.StoreSaScale)) {
    macSel := workPendingOp
  }

  val macOut      = Wire(Vec(lanes, accType.cloneType))
  val cmpOut      = Wire(Vec(lanes, accType.cloneType))
  val writeOpData = WireDefault(zeroLine)
  val writeEn     = WireDefault(false.B)
  val writeRow    = WireDefault(io.writeRow)

  for (lane <- 0 until lanes) {
    val macAcc = WireDefault(zeroElem)
    val macM1  = WireDefault(zeroElem)
    val macM2  = WireDefault(zeroElem)

    cmpOut(lane) := Mux(io.saIn(lane) > reduceReg(lane), io.saIn(lane), reduceReg(lane))

    switch(macSel) {
      is(AccOp.StoreSaReduceSum) {
        macAcc := reduceReg(lane)
        macM1  := io.saIn(lane)
        macM2  := reduceReg(lane).identity
      }
      is(AccOp.SubRowReduceToScale) {
        macAcc := readData(lane)
        macM1  := reduceReg(lane).neg
        macM2  := reduceReg(lane).identity
      }
      is(AccOp.StoreSaScale) {
        macAcc := saInPipe(lane)
        macM1  := readData(lane)
        macM2  := scaleReg(lane)
      }
    }

    macOut(lane) := macAcc.mac(macM1, macM2)
  }

  switch(io.op) {
    is(AccOp.Reset) {
      reduceReg  := zeroLine
      scaleReg   := zeroLine
      sramOutReg := zeroLine
    }
    is(AccOp.StoreSa) {
      writeEn     := true.B
      writeOpData := io.saIn
    }
    is(AccOp.StoreSaReduceMax) {
      reduceReg   := cmpOut
      writeEn     := true.B
      writeOpData := io.saIn
    }
    is(AccOp.StoreSaReduceSum) {
      reduceReg   := macOut
      writeEn     := true.B
      writeOpData := io.saIn
    }
    is(AccOp.StoreReduce) {
      writeEn     := true.B
      writeOpData := reduceReg
    }
    is(AccOp.StoreScale) {
      writeEn     := true.B
      writeOpData := scaleReg
    }
    is(AccOp.ClearReduceZero) {
      reduceReg := zeroLine
    }
    is(AccOp.ClearReduceMin) {
      reduceReg := minLine
    }
    is(AccOp.ClearScaleZero) {
      scaleReg := zeroLine
    }
    is(AccOp.ClearScaleMin) {
      scaleReg := minLine
    }
  }

  when(workReadPending) {
    switch(workPendingOp) {
      is(AccOp.LoadReduce) {
        reduceReg := readData
      }
      is(AccOp.LoadScale) {
        scaleReg := readData
      }
      is(AccOp.SubRowReduceToScale) {
        scaleReg := macOut
      }
      is(AccOp.StoreSaScale) {
        writeEn     := true.B
        writeRow    := workRowPipe
        writeOpData := macOut
      }
    }
  }

  assert(
    !(workReadPending && workPendingOp === AccOp.StoreSaScale && workWriteIssue),
    "Accumulator has one SRAM write port; do not overlap StoreSaScale completion with another store op"
  )

  for (lane <- 0 until lanes) {
    when(writeEn) {
      banks(lane).write(writeRow, writeOpData(lane))
    }
  }

  io.saOut := sramOutReg
  io.spOut := sramOutReg
}

class PinnacleMemoryHostIO[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val rdAddr = Input(UInt(addrWidth.W))
  val rdData = Output(Vec(portWidth, dataType))
  val wrEn   = Input(Bool())
  val wrAddr = Input(UInt(addrWidth.W))
  val wrData = Input(Vec(portWidth, dataType))
}

object PinnacleWestSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val Acc = 2.U(2.W); val External = 3.U(2.W)
}

object PinnacleNorthSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val Acc = 2.U(2.W); val External = 3.U(2.W)
}

object PinnacleAccInSrc {
  val Zero = 0.U(2.W); val ArrayGemm = 1.U(2.W); val ArrayPinn = 2.U(2.W); val External = 3.U(2.W)
}

object PinnacleSpWriteSrc {
  val External = 0.U(2.W); val Acc = 1.U(2.W)
}

class PinnacleSystem[T <: Data: Arithmetic](
    N: Int,
    dataType: T,
    accType: T,
    scratchpadDepth: Int,
    stripHeight: Int,
    accumulatorDepth: Int = 0
) extends Module {
  private val resolvedAccumulatorDepth = if (accumulatorDepth > 0) accumulatorDepth else N
  require(N > 0 && scratchpadDepth >= N && resolvedAccumulatorDepth > 0, "invalid PinnacleSystem dimensions")
  PinnacleConfig.requireSupportedGeometry(N, stripHeight)

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val spAddrWidth = SramMacroConfig.scratchpadAddrWidth(N, scratchpadDepth)
  private val accRowIdxW  = math.max(1, log2Ceil(resolvedAccumulatorDepth))
  private val zeroDataLine = 0.U.asTypeOf(Vec(N, dataType.cloneType))
  private val zeroAccLine  = 0.U.asTypeOf(Vec(N, accType.cloneType))

  val io = IO(new Bundle {
    val sp = new PinnacleMemoryHostIO(spAddrWidth, N, dataType)
    val ctrl = Input(new Bundle {
      val spWestReadEn, spNorthReadEn, spWriteEn = Bool()
      val spWestReadAddr, spNorthReadAddr, spWriteAddr = UInt(spAddrWidth.W)
      val spWriteSrc = UInt(2.W)
      val spWriteData = Vec(N, dataType.cloneType)

      val westSrc      = UInt(2.W)
      val westOp       = Vec(N, WestOp())
      val westExternal = Vec(N, dataType.cloneType)

      val northSrc      = UInt(2.W)
      val northOp       = Vec(N, NorthOp())
      val northExternal = Vec(N, accType.cloneType)

      val accInSrc      = UInt(2.W)
      val accExternal   = Vec(N, accType.cloneType)
      val accOp         = AccOp()
      val accStreamReadRow, accOpReadRow, accWriteRow = UInt(accRowIdxW.W)
    })
  })

  val scratchpad = Scratchpad(N, scratchpadDepth, 2, dataType)
  val systolic   = Module(new SystolicArray(N, dataType, accType, stripHeight))
  val accum      = Module(new Accumulator(accType, N, resolvedAccumulatorDepth))

  val spWestData  = scratchpad.io.ctrlReadData(0)
  val spNorthData = VecInit(Seq.tabulate(N)(lane => scratchpad.io.ctrlReadData(1)(lane).withWidthOf(accType)))
  val accSaData   = accum.io.saOut
  val accSaAsData = VecInit(accSaData.map(_.withWidthOf(dataType)))
  val accSpAsData = VecInit(accum.io.spOut.map(_.withWidthOf(dataType)))

  val westData = MuxLookup(io.ctrl.westSrc, zeroDataLine)(
    Seq(
      PinnacleWestSrc.Scratchpad -> spWestData,
      PinnacleWestSrc.Acc        -> accSaAsData,
      PinnacleWestSrc.External   -> io.ctrl.westExternal
    )
  )
  val northData = MuxLookup(io.ctrl.northSrc, zeroAccLine)(
    Seq(
      PinnacleNorthSrc.Scratchpad -> spNorthData,
      PinnacleNorthSrc.Acc        -> accSaData,
      PinnacleNorthSrc.External   -> io.ctrl.northExternal
    )
  )
  val accInData = MuxLookup(io.ctrl.accInSrc, zeroAccLine)(
    Seq(
      PinnacleAccInSrc.ArrayGemm -> systolic.io.gemmSouth,
      PinnacleAccInSrc.ArrayPinn -> systolic.io.pinnSouth,
      PinnacleAccInSrc.External  -> io.ctrl.accExternal
    )
  )
  val spWriteData = MuxLookup(io.ctrl.spWriteSrc, io.ctrl.spWriteData)(
    Seq(PinnacleSpWriteSrc.Acc -> accSpAsData)
  )

  scratchpad.io.host.rd(0).addr := io.sp.rdAddr
  scratchpad.io.host.wr(0).en   := io.sp.wrEn
  scratchpad.io.host.wr(0).addr := io.sp.wrAddr
  scratchpad.io.host.wr(0).data := io.sp.wrData
  scratchpad.io.ctrlReadEn(0)   := io.ctrl.spWestReadEn
  scratchpad.io.ctrlReadAddr(0) := io.ctrl.spWestReadAddr
  scratchpad.io.ctrlReadEn(1)   := io.ctrl.spNorthReadEn
  scratchpad.io.ctrlReadAddr(1) := io.ctrl.spNorthReadAddr
  scratchpad.io.ctrlWriteEn     := io.ctrl.spWriteEn
  scratchpad.io.ctrlWriteAddr   := io.ctrl.spWriteAddr
  scratchpad.io.ctrlWriteData   := spWriteData
  io.sp.rdData                  := scratchpad.io.host.rd(0).data

  systolic.io.west := VecInit(Seq.tabulate(N) { row =>
    val token = Wire(new WestToken(dataType))
    token.op   := io.ctrl.westOp(row)
    token.data := westData(row)
    token
  })
  systolic.io.north := VecInit(Seq.tabulate(N) { lane =>
    val token = Wire(new NorthToken(accType))
    token.op   := io.ctrl.northOp(lane)
    token.data := northData(lane)
    token
  })

  accum.io.op            := io.ctrl.accOp
  accum.io.saIn          := accInData
  accum.io.streamReadRow := io.ctrl.accStreamReadRow
  accum.io.opReadRow     := io.ctrl.accOpReadRow
  accum.io.writeRow      := io.ctrl.accWriteRow
}
