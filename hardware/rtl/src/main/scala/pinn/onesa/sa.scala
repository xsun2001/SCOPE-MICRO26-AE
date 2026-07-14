package pinn.onesa

import chisel3._
import chisel3.util._
import pinn.common._

object OneSaArrayOp extends ChiselEnum {
  val Hold, LoadWeight, Compute, Nonlinear = Value
}

class OneSaSimdPayload[T <: Data](private val simdWidth: Int, private val gen: T) extends Bundle {
  val lanes = Vec(simdWidth, gen.cloneType)

  def laneCount: Int = simdWidth
}

object OneSaSimdPayload {
  def zero[T <: Data](simdWidth: Int, dataType: T): OneSaSimdPayload[T] =
    0.U.asTypeOf(new OneSaSimdPayload(simdWidth, dataType))
}

class OneSaArrayIO[T <: Data: Arithmetic](N: Int, simdWidth: Int, dataType: T, accType: T) extends Bundle {
  val op = Input(OneSaArrayOp())

  val west = Input(Vec(N, new OneSaSimdPayload(simdWidth, dataType)))
  val north = Input(Vec(N, new OneSaSimdPayload(simdWidth, dataType)))
  val psum = Input(Vec(N, accType))

  val south = Output(Vec(N, accType))

  val diag = Output(Vec(N, accType))
  val diagPairs = Output(Vec(N, Vec(simdWidth / 2, accType)))
  val diagValid = Output(Vec(N, Bool()))
}

class OneSaPEIO[T <: Data](simdWidth: Int, dataType: T, accType: T) extends Bundle {
  val op = Input(OneSaArrayOp())

  val westIn = Input(new OneSaSimdPayload(simdWidth, dataType))
  val northIn = Input(new OneSaSimdPayload(simdWidth, dataType))
  val psumIn = Input(accType)

  val eastOut = Output(new OneSaSimdPayload(simdWidth, dataType))
  val southOut = Output(new OneSaSimdPayload(simdWidth, dataType))
  val psumOut = Output(accType)

  val diagPairsOut = Output(Vec(simdWidth / 2, accType))
}

class OneSaPE[T <: Data: Arithmetic](simdWidth: Int, dataType: T, accType: T, isDiagonalPe: Boolean)
    extends Module {
  override def desiredName: String = if (isDiagonalPe) "OneSaPEComputation" else "OneSaPETransmission"

  require(simdWidth > 0, s"simdWidth must be positive, got $simdWidth")
  require(simdWidth % 2 == 0, s"OneSa requires an even simdWidth, got $simdWidth")

  val io = IO(new OneSaPEIO(simdWidth, dataType, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val zeroPayload = OneSaSimdPayload.zero(simdWidth, dataType)
  val zeroAcc = 0.U.asTypeOf(accType)
  val zeroPairs = 0.U.asTypeOf(Vec(simdWidth / 2, accType.cloneType))

  val regEast = RegInit(zeroPayload)
  val regSouth = RegInit(zeroPayload)
  val regPsum = RegInit(zeroAcc)
  val regWeight = RegInit(zeroPayload)
  val regSharedPairs = RegInit(zeroPairs)

  private def reduceAdd(values: Seq[T]): T =
    values.reduceOption((lhs, rhs) => (lhs + rhs).withWidthOf(accType)).getOrElse(zeroAcc)

  private def pairwiseMac(lhs: Seq[T], rhs: Seq[T]): Vec[T] = {
    val products = lhs.zip(rhs).map { case (a, b) =>
      (a.withWidthOf(accType) * b.withWidthOf(accType)).withWidthOf(accType)
    }
    val pairs = products
      .grouped(2)
      .map(group => reduceAdd(group).withWidthOf(accType))
      .toSeq
    VecInit(pairs)
  }

  val sharedMacRhs = Mux(io.op === OneSaArrayOp.Compute, regWeight, io.northIn)
  val sharedPairs = pairwiseMac(io.westIn.lanes.toSeq, sharedMacRhs.lanes.toSeq)
  val linearDot = reduceAdd(regSharedPairs.toSeq)

  io.eastOut := regEast
  io.southOut := regSouth
  io.psumOut := regPsum
  io.diagPairsOut := Mux(isDiagonalPe.B && io.op === OneSaArrayOp.Nonlinear, sharedPairs, zeroPairs)

  regEast := zeroPayload
  regSouth := zeroPayload
  regPsum := zeroAcc

  when(io.op === OneSaArrayOp.LoadWeight) {
    regWeight := io.westIn
  }

  switch(io.op) {
    is(OneSaArrayOp.LoadWeight) {
      regEast := io.westIn
      regPsum := io.psumIn
    }
    is(OneSaArrayOp.Compute) {
      regEast := io.westIn
      regSharedPairs := sharedPairs
      regPsum := io.psumIn.mac(linearDot, dataType.identity.withWidthOf(accType))
    }
    is(OneSaArrayOp.Nonlinear) {
      when(!isDiagonalPe.B) {
        regEast := io.westIn
        regSouth := io.northIn
      }
    }
  }
}

class OneSaMesh[T <: Data: Arithmetic](N: Int, simdWidth: Int, dataType: T, accType: T) extends Module {
  val io = IO(new Bundle {
    val op = Input(Vec(N, OneSaArrayOp()))
    val west = Input(Vec(N, new OneSaSimdPayload(simdWidth, dataType)))
    val north = Input(Vec(N, new OneSaSimdPayload(simdWidth, dataType)))
    val psum = Input(Vec(N, accType))

    val south = Output(Vec(N, accType))
    val diag = Output(Vec(N, accType))
    val diagPairs = Output(Vec(N, Vec(simdWidth / 2, accType)))
    val diagValid = Output(Vec(N, Bool()))
  })

  val pes = Seq.tabulate(N, N) { (row, col) =>
    Module(new OneSaPE(simdWidth, dataType, accType, isDiagonalPe = row == col))
  }

  for {
    row <- 0 until N
    col <- 0 until N
  } {
    pes(row)(col).io.op := io.op(row)
    pes(row)(col).io.westIn := (if (col == 0) io.west(row) else pes(row)(col - 1).io.eastOut)
    pes(row)(col).io.northIn := (if (row == 0) io.north(col) else pes(row - 1)(col).io.southOut)
    pes(row)(col).io.psumIn := (if (row == 0) io.psum(col) else pes(row - 1)(col).io.psumOut)
  }

  for (col <- 0 until N) {
    io.south(col) := pes(N - 1)(col).io.psumOut
  }

  for (idx <- 0 until N) {
    io.diag(idx) := pes(idx)(idx).io.diagPairsOut(0)
    io.diagPairs(idx) := pes(idx)(idx).io.diagPairsOut
    io.diagValid(idx) := io.op(idx) === OneSaArrayOp.Nonlinear
  }
}

class OneSaBuffer[T <: Data](N: Int, dataType: T, delayFn: Int => Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(N, dataType))
    val out = Output(Vec(N, dataType))
  })

  for (idx <- 0 until N) {
    io.out(idx) := ShiftRegister(io.in(idx), delayFn(idx))
  }
}

object OneSaBuffer {
  def apply[T <: Data](in: Vec[T], delayFn: Int => Int): Vec[T] = {
    val buffer = Module(new OneSaBuffer(in.length, in(0).cloneType, delayFn))
    buffer.io.in := in
    buffer.io.out
  }
}

class OneSaArray[T <: Data: Arithmetic](N: Int, simdWidth: Int, dataType: T, accType: T) extends Module {
  require(simdWidth > 0, s"simdWidth must be positive, got $simdWidth")
  require(simdWidth % 2 == 0, s"OneSa requires an even simdWidth, got $simdWidth")

  val io = IO(new OneSaArrayIO(N, simdWidth, dataType, accType))

  val mesh = Module(new OneSaMesh(N, simdWidth, dataType, accType))

  mesh.io.op := OneSaBuffer(VecInit.fill(N)(io.op), row => row + 1)
  mesh.io.west := OneSaBuffer(io.west, row => row + 1)
  mesh.io.north := OneSaBuffer(io.north, col => col + 1)
  mesh.io.psum := OneSaBuffer(io.psum, col => col + 1)

  io.south := OneSaBuffer(mesh.io.south, col => N - col - 1)
  io.diag := OneSaBuffer(mesh.io.diag, idx => 2 * (N - idx - 1))
  io.diagPairs := OneSaBuffer(mesh.io.diagPairs, idx => 2 * (N - idx - 1))
  io.diagValid := OneSaBuffer(mesh.io.diagValid, idx => 2 * (N - idx - 1) + 1)
}

class OneSaL3RearrangeIO(N: Int, dataWidth: Int, segmentCount: Int, simdWidth: Int) extends Bundle {
  private val idxWidth = math.max(1, log2Ceil(segmentCount))
  private val shiftWidth = math.max(1, log2Ceil(dataWidth + 1))
  private val segmentCalcBits = math.max(idxWidth + 2, shiftWidth + 1)

  val in = Input(Vec(N, SInt(dataWidth.W)))

  val segmentShift = Input(UInt(shiftWidth.W))
  val segmentOffset = Input(SInt(segmentCalcBits.W))

  val kWriteEn = Input(Bool())
  val kWriteAddr = Input(UInt(idxWidth.W))
  val kWriteData = Input(SInt(dataWidth.W))
  val bWriteEn = Input(Bool())
  val bWriteAddr = Input(UInt(idxWidth.W))
  val bWriteData = Input(SInt(dataWidth.W))

  val segmentId = Output(Vec(N, UInt(idxWidth.W)))
  val westOut = Output(Vec(N, new OneSaSimdPayload(simdWidth, SInt(dataWidth.W))))
  val northOut = Output(Vec(N, new OneSaSimdPayload(simdWidth, SInt(dataWidth.W))))
  val tableKDebug = Output(Vec(segmentCount, SInt(dataWidth.W)))
  val tableBDebug = Output(Vec(segmentCount, SInt(dataWidth.W)))
}

class OneSaL3Rearrange(N: Int, dataWidth: Int, segmentCount: Int, simdWidth: Int = 2) extends Module {
  require(segmentCount > 0, s"segmentCount must be positive, got $segmentCount")
  require(simdWidth >= 2, s"simdWidth must be at least 2, got $simdWidth")
  require(simdWidth % 2 == 0, s"OneSa requires an even simdWidth, got $simdWidth")

  val io = IO(new OneSaL3RearrangeIO(N, dataWidth, segmentCount, simdWidth))

  private val idxWidth = math.max(1, log2Ceil(segmentCount))
  private val one = 1.S(dataWidth.W)
  private val zeroPayload = OneSaSimdPayload.zero(simdWidth, SInt(dataWidth.W))
  private val kTableReg = RegInit(VecInit(Seq.fill(segmentCount)(0.S(dataWidth.W))))
  private val bTableReg = RegInit(VecInit(Seq.fill(segmentCount)(0.S(dataWidth.W))))

  when(io.kWriteEn) {
    assert(io.kWriteAddr < segmentCount.U, s"k-table write address must be < $segmentCount")
    when(io.kWriteAddr < segmentCount.U) {
      kTableReg(io.kWriteAddr) := io.kWriteData
    }
  }

  when(io.bWriteEn) {
    assert(io.bWriteAddr < segmentCount.U, s"b-table write address must be < $segmentCount")
    when(io.bWriteAddr < segmentCount.U) {
      bTableReg(io.bWriteAddr) := io.bWriteData
    }
  }

  for (idx <- 0 until N) {
    val shifted = (io.in(idx) >> io.segmentShift).asSInt + io.segmentOffset
    val clamped = Wire(SInt((idxWidth + 2).W))

    when(shifted < 0.S) {
      clamped := 0.S
    }.elsewhen(shifted > (segmentCount - 1).S) {
      clamped := (segmentCount - 1).S
    }.otherwise {
      clamped := shifted
    }

    val segmentIdx = clamped.asUInt.apply(idxWidth - 1, 0)

    io.segmentId(idx) := segmentIdx
    io.westOut(idx) := zeroPayload
    io.northOut(idx) := zeroPayload
    io.westOut(idx).lanes(0) := io.in(idx)
    io.westOut(idx).lanes(1) := one
    io.northOut(idx).lanes(0) := kTableReg(segmentIdx)
    io.northOut(idx).lanes(1) := bTableReg(segmentIdx)
  }

  io.tableKDebug := kTableReg
  io.tableBDebug := bTableReg
}

object OneSaAccOp extends ChiselEnum {
  val Nop, Reset, LoadRow, StoreRow, AccRow = Value
}

class OneSaAccumulatorIO[T <: Data](lanes: Int, depth: Int, accType: T) extends Bundle {
  private val rowIdxW = math.max(1, log2Ceil(depth))

  val op = Input(OneSaAccOp())
  val saIn = Input(Vec(lanes, accType))
  val readRow = Input(UInt(rowIdxW.W))
  val writeRow = Input(UInt(rowIdxW.W))

  val saOut = Output(Vec(lanes, accType))
  val spOut = Output(Vec(lanes, accType))
}

class OneSaAccumulator[T <: Data: Arithmetic](accType: T, lanes: Int, depth: Int) extends Module {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(depth > 0, s"depth must be positive, got $depth")

  val io = IO(new OneSaAccumulatorIO(lanes, depth, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val zeroLine = 0.U.asTypeOf(Vec(lanes, accType.cloneType))
  private val rowIdxW = math.max(1, log2Ceil(depth))

  val banks = Seq.fill(lanes)(SyncReadMem(depth, accType.cloneType))

  val sramOutReg = RegInit(zeroLine)
  val pendingReadOp = RegInit(OneSaAccOp.Nop)
  val pendingWriteRow = RegInit(0.U(rowIdxW.W))
  val pendingSaIn = RegInit(zeroLine)

  val readIssue = io.op === OneSaAccOp.LoadRow || io.op === OneSaAccOp.AccRow
  val readAddr = Mux(io.op === OneSaAccOp.AccRow, io.writeRow, io.readRow)
  val readData = Wire(Vec(lanes, accType.cloneType))
  for (lane <- 0 until lanes) {
    readData(lane) := banks(lane).read(readAddr, readIssue)
  }

  val writeOpData = WireDefault(zeroLine)
  val writeEn = WireDefault(false.B)
  val writeRow = WireDefault(io.writeRow)
  val accWriteData = Wire(Vec(lanes, accType.cloneType))
  for (lane <- 0 until lanes) {
    accWriteData(lane) := (readData(lane) + pendingSaIn(lane)).withWidthOf(accType)
  }

  when(pendingReadOp === OneSaAccOp.LoadRow) {
    sramOutReg := readData
  }.elsewhen(pendingReadOp === OneSaAccOp.AccRow) {
    writeEn := true.B
    writeRow := pendingWriteRow
    writeOpData := accWriteData
  }

  switch(io.op) {
    is(OneSaAccOp.Reset) {
      sramOutReg := zeroLine
      pendingReadOp := OneSaAccOp.Nop
    }
    is(OneSaAccOp.StoreRow) {
      writeEn := true.B
      writeOpData := io.saIn
    }
  }

  assert(
    !(pendingReadOp === OneSaAccOp.AccRow && (io.op === OneSaAccOp.StoreRow || io.op === OneSaAccOp.Reset)),
    "Accumulator has one write port; caller must not overlap AccRow completion with another write op"
  )

  pendingReadOp := Mux(readIssue, io.op, OneSaAccOp.Nop)
  when(readIssue) {
    pendingWriteRow := io.writeRow
    pendingSaIn := io.saIn
  }

  for (lane <- 0 until lanes) {
    when(writeEn) {
      banks(lane).write(writeRow, writeOpData(lane))
    }
  }

  io.saOut := sramOutReg
  io.spOut := sramOutReg
}

class OneSaMemoryHostIO[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val rdAddr = Input(UInt(addrWidth.W))
  val rdData = Output(Vec(portWidth, dataType))
  val wrEn = Input(Bool())
  val wrAddr = Input(UInt(addrWidth.W))
  val wrData = Input(Vec(portWidth, dataType))
}

object OneSaRearrangeInSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val Acc = 2.U(2.W); val External = 3.U(2.W)
}

object OneSaWestSrc {
  val Zero = 0.U(2.W); val Rearrange = 1.U(2.W); val Scratchpad = 2.U(2.W); val External = 3.U(2.W)
}

object OneSaNorthSrc {
  val Zero = 0.U(2.W); val Rearrange = 1.U(2.W); val Scratchpad = 2.U(2.W); val External = 3.U(2.W)
}

object OneSaPsumSrc {
  val Zero = 0.U(2.W); val Acc = 1.U(2.W); val External = 2.U(2.W)
}

object OneSaAccInSrc {
  val Zero = 0.U(3.W); val ArraySouth = 1.U(3.W); val ArrayDiag = 2.U(3.W); val Scratchpad = 3.U(3.W); val External = 4.U(3.W)
}

object OneSaSpWriteSrc {
  val External = 0.U(2.W); val Acc = 1.U(2.W)
}

class OneSaSystem(
    N: Int,
    dataWidth: Int,
    accWidth: Int,
    segmentCount: Int,
    simdWidth: Int = 2,
    scratchpadDepth: Int = 64,
    accumulatorDepth: Int = 0
) extends Module {
  require(N > 0, s"OneSaSystem requires N > 0, got $N")
  require(segmentCount > 0, s"segmentCount must be positive, got $segmentCount")
  require(simdWidth >= 2, s"simdWidth must be at least 2, got $simdWidth")
  require(simdWidth % 2 == 0, s"OneSa requires an even simdWidth, got $simdWidth")
  require(scratchpadDepth > 0, s"scratchpadDepth must be positive, got $scratchpadDepth")

  private val resolvedAccumulatorDepth = if (accumulatorDepth > 0) accumulatorDepth else N
  private val dataType = SInt(dataWidth.W)
  private val accType = SInt(accWidth.W)
  private val spAddrWidth = SramMacroConfig.scratchpadAddrWidth(N, scratchpadDepth)
  private val shiftWidth = math.max(1, log2Ceil(dataWidth + 1))
  private val segmentCalcBits = math.max(log2Ceil(segmentCount) + 2, shiftWidth + 1)
  private val tableIdxWidth = math.max(1, log2Ceil(segmentCount))
  private val accRowIdxW = math.max(1, log2Ceil(resolvedAccumulatorDepth))

  val ev = implicitly[Arithmetic[SInt]]
  import ev._

  private val zeroDataLine = 0.U.asTypeOf(Vec(N, dataType.cloneType))
  private val zeroAccLine = 0.U.asTypeOf(Vec(N, accType.cloneType))
  private val zeroSimdLine = 0.U.asTypeOf(Vec(N, new OneSaSimdPayload(simdWidth, dataType)))

  val io = IO(new Bundle {
    val sp = new OneSaMemoryHostIO(spAddrWidth, N, dataType)
    val ctrl = Input(new Bundle {
      val spReadEn = Bool()
      val spReadAddr = UInt(spAddrWidth.W)
      val spWriteEn = Bool()
      val spWriteAddr = UInt(spAddrWidth.W)
      val spWriteSrc = UInt(2.W)
      val spWriteData = Vec(N, dataType.cloneType)

      val rearrangeInSrc = UInt(2.W)
      val rearrangeExternal = Vec(N, dataType.cloneType)
      val segmentShift = UInt(shiftWidth.W)
      val segmentOffset = SInt(segmentCalcBits.W)
      val kTableWriteEn = Bool()
      val kTableWriteAddr = UInt(tableIdxWidth.W)
      val kTableWriteData = SInt(dataWidth.W)
      val bTableWriteEn = Bool()
      val bTableWriteAddr = UInt(tableIdxWidth.W)
      val bTableWriteData = SInt(dataWidth.W)

      val arrayOp = OneSaArrayOp()
      val westSrc = UInt(2.W)
      val westExternal = Vec(N, new OneSaSimdPayload(simdWidth, dataType))
      val northSrc = UInt(2.W)
      val northExternal = Vec(N, new OneSaSimdPayload(simdWidth, dataType))
      val psumSrc = UInt(2.W)
      val psumExternal = Vec(N, accType.cloneType)

      val accInSrc = UInt(3.W)
      val accExternal = Vec(N, accType.cloneType)
      val accOp = OneSaAccOp()
      val accReadRow = UInt(accRowIdxW.W)
      val accWriteRow = UInt(accRowIdxW.W)
    })
  })

  val scratchpad = Scratchpad(N, scratchpadDepth, 1, dataType)
  val rearrange = Module(new OneSaL3Rearrange(N, dataWidth, segmentCount, simdWidth))
  val systolic = Module(new OneSaArray(N, simdWidth, dataType, accType))
  val accum = Module(new OneSaAccumulator(accType, N, resolvedAccumulatorDepth))

  private def lineToPayload(line: Vec[SInt]): Vec[OneSaSimdPayload[SInt]] = {
    val payload = WireDefault(zeroSimdLine)
    for (idx <- 0 until N) {
      payload(idx).lanes(0) := line(idx)
    }
    payload
  }

  val spReadData = scratchpad.io.ctrlReadData(0)
  val spReadAsAcc = VecInit(spReadData.map(_.withWidthOf(accType)))
  val spReadAsPayload = lineToPayload(spReadData)
  val accSaAsData = VecInit(accum.io.saOut.map(_.withWidthOf(dataType)))
  val accSpAsData = VecInit(accum.io.spOut.map(_.withWidthOf(dataType)))

  val rearrangeInput = MuxLookup(io.ctrl.rearrangeInSrc, zeroDataLine)(
    Seq(
      OneSaRearrangeInSrc.Scratchpad -> spReadData,
      OneSaRearrangeInSrc.Acc -> accSaAsData,
      OneSaRearrangeInSrc.External -> io.ctrl.rearrangeExternal
    )
  )
  val westData = MuxLookup(io.ctrl.westSrc, zeroSimdLine)(
    Seq(
      OneSaWestSrc.Rearrange -> rearrange.io.westOut,
      OneSaWestSrc.Scratchpad -> spReadAsPayload,
      OneSaWestSrc.External -> io.ctrl.westExternal
    )
  )
  val northData = MuxLookup(io.ctrl.northSrc, zeroSimdLine)(
    Seq(
      OneSaNorthSrc.Rearrange -> rearrange.io.northOut,
      OneSaNorthSrc.Scratchpad -> spReadAsPayload,
      OneSaNorthSrc.External -> io.ctrl.northExternal
    )
  )
  val psumData = MuxLookup(io.ctrl.psumSrc, zeroAccLine)(
    Seq(
      OneSaPsumSrc.Acc -> accum.io.saOut,
      OneSaPsumSrc.External -> io.ctrl.psumExternal
    )
  )
  val accInData = MuxLookup(io.ctrl.accInSrc, zeroAccLine)(
    Seq(
      OneSaAccInSrc.ArraySouth -> systolic.io.south,
      OneSaAccInSrc.ArrayDiag -> systolic.io.diag,
      OneSaAccInSrc.Scratchpad -> spReadAsAcc,
      OneSaAccInSrc.External -> io.ctrl.accExternal
    )
  )
  val spWriteData = MuxLookup(io.ctrl.spWriteSrc, io.ctrl.spWriteData)(
    Seq(OneSaSpWriteSrc.Acc -> accSpAsData)
  )

  scratchpad.io.host.rd(0).addr := io.sp.rdAddr
  scratchpad.io.host.wr(0).en := io.sp.wrEn
  scratchpad.io.host.wr(0).addr := io.sp.wrAddr
  scratchpad.io.host.wr(0).data := io.sp.wrData
  scratchpad.io.ctrlReadEn(0) := io.ctrl.spReadEn
  scratchpad.io.ctrlReadAddr(0) := io.ctrl.spReadAddr
  scratchpad.io.ctrlWriteEn := io.ctrl.spWriteEn
  scratchpad.io.ctrlWriteAddr := io.ctrl.spWriteAddr
  scratchpad.io.ctrlWriteData := spWriteData
  io.sp.rdData := scratchpad.io.host.rd(0).data

  rearrange.io.in := rearrangeInput
  rearrange.io.segmentShift := io.ctrl.segmentShift
  rearrange.io.segmentOffset := io.ctrl.segmentOffset
  rearrange.io.kWriteEn := io.ctrl.kTableWriteEn
  rearrange.io.kWriteAddr := io.ctrl.kTableWriteAddr
  rearrange.io.kWriteData := io.ctrl.kTableWriteData
  rearrange.io.bWriteEn := io.ctrl.bTableWriteEn
  rearrange.io.bWriteAddr := io.ctrl.bTableWriteAddr
  rearrange.io.bWriteData := io.ctrl.bTableWriteData

  systolic.io.op := io.ctrl.arrayOp
  systolic.io.west := westData
  systolic.io.north := northData
  systolic.io.psum := psumData

  accum.io.op := io.ctrl.accOp
  accum.io.saIn := accInData
  accum.io.readRow := io.ctrl.accReadRow
  accum.io.writeRow := io.ctrl.accWriteRow
}
