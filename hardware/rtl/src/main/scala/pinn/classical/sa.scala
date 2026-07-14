package pinn.classical

import chisel3._
import chisel3.util._
import pinn.common._

object ClassicalWestOp extends ChiselEnum {
  val Hold, Preload, Compute = Value
}

object ClassicalAccOp extends ChiselEnum {
  val Nop, Load, Store, Accumulate = Value
}

class ClassicalWestToken[T <: Data](dataType: T) extends Bundle {
  val op   = ClassicalWestOp()
  val data = dataType.cloneType
}

class PEIO[T <: Data](dataType: T, accType: T) extends Bundle {
  val westOpIn = Input(ClassicalWestOp())
  val westIn   = Input(dataType)
  val northIn  = Input(accType)
  val eastOut  = Output(dataType)
  val southOut = Output(accType)
}

class PE[T <: Data: Arithmetic](dataType: T, accType: T) extends Module {
  val io = IO(new PEIO(dataType, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  val regE = RegInit(0.U.asTypeOf(dataType.cloneType))
  val regS = RegInit(0.U.asTypeOf(accType.cloneType))
  val regW = RegInit(0.U.asTypeOf(dataType.cloneType))

  val macOut = io.northIn.mac(regW.withWidthOf(accType), io.westIn.withWidthOf(accType))

  io.eastOut  := Mux(io.westOpIn === ClassicalWestOp.Preload, regW, regE)
  io.southOut := regS

  switch(io.westOpIn) {
    is(ClassicalWestOp.Preload) {
      regW := io.westIn
    }
    is(ClassicalWestOp.Compute) {
      regE := io.westIn
      regS := macOut
    }
  }
}

class ClassicalBuffer[T <: Data](N: Int, dataType: T, delayFn: Int => Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(N, dataType))
    val out = Output(Vec(N, dataType))
  })

  for (idx <- 0 until N) {
    io.out(idx) := ShiftRegister(io.in(idx), delayFn(idx))
  }
}

object ClassicalBuffer {
  def apply[T <: Data](in: Vec[T], delayFn: Int => Int): Vec[T] = {
    val buffer = Module(new ClassicalBuffer(in.length, in(0).cloneType, delayFn))
    buffer.io.in := in
    buffer.io.out
  }
}

class SystolicArrayIO[T <: Data](N: Int, dataType: T, accType: T) extends Bundle {
  val west       = Input(Vec(N, new ClassicalWestToken(dataType)))
  val north      = Input(Vec(N, accType))
  val south      = Output(Vec(N, accType))
  val southValid = Output(Bool())
}

class SystolicArray[T <: Data: Arithmetic](N: Int, dataType: T, accType: T) extends Module {
  require(N > 0, s"classical direct system requires positive N, got $N")
  private val gemmLatency = 2 * N - 1
  private val inputBaseLatency = 1

  val io = IO(new SystolicArrayIO(N, dataType, accType))

  val bufferedWest  = ClassicalBuffer(io.west, row => row + inputBaseLatency)
  val bufferedNorth = ClassicalBuffer(io.north, col => col + inputBaseLatency)
  val meshSouth     = Wire(Vec(N, Vec(N, accType.cloneType)))
  val bottomSouth   = Wire(Vec(N, accType.cloneType))

  for (row <- 0 until N) {
    val pes = Seq.fill(N)(Module(new PE(dataType, accType)))
    for (col <- 0 until N) {
      val westIn  = if (col == 0) bufferedWest(row).data else pes(col - 1).io.eastOut
      val northIn = if (row == 0) bufferedNorth(col) else meshSouth(row - 1)(col)

      pes(col).io.westOpIn := bufferedWest(row).op
      pes(col).io.westIn   := westIn
      pes(col).io.northIn  := northIn

      meshSouth(row)(col) := pes(col).io.southOut
    }
  }

  for (col <- 0 until N) {
    bottomSouth(col) := meshSouth(N - 1)(col)
  }

  io.south := ClassicalBuffer(bottomSouth, col => N - col - 1)
  io.southValid := ShiftRegister(
    io.west.map(_.op === ClassicalWestOp.Compute).reduce(_ || _),
    gemmLatency + inputBaseLatency,
    false.B
  )
}

class AccumulatorIO[T <: Data](lanes: Int, depth: Int, accType: T) extends Bundle {
  private val rowIdxW = math.max(1, log2Ceil(depth))

  val op       = Input(ClassicalAccOp())
  val saIn     = Input(Vec(lanes, accType))
  val readRow  = Input(UInt(rowIdxW.W))
  val writeRow = Input(UInt(rowIdxW.W))
  val saOut    = Output(Vec(lanes, accType))
  val spOut    = Output(Vec(lanes, accType))
}

class Accumulator[T <: Data: Arithmetic](accType: T, lanes: Int, depth: Int) extends Module {
  require(lanes > 0, s"lanes must be positive, got $lanes")
  require(depth > 0, s"depth must be positive, got $depth")

  val io = IO(new AccumulatorIO(lanes, depth, accType))

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val zeroLine = 0.U.asTypeOf(Vec(lanes, accType.cloneType))
  private val banks    = Seq.fill(lanes)(SyncReadMem(depth, accType.cloneType))

  val sramOutReg = RegInit(zeroLine)

  val readIssue     = io.op === ClassicalAccOp.Load
  val accumIssue    = io.op === ClassicalAccOp.Accumulate
  val sramReadIssue = readIssue || accumIssue

  val sramReadData = Wire(Vec(lanes, accType.cloneType))
  for (lane <- 0 until lanes) {
    sramReadData(lane) := banks(lane).read(io.readRow, sramReadIssue)
  }

  val readCaptureEn = RegNext(readIssue, false.B)
  when(readCaptureEn) {
    sramOutReg := sramReadData
  }

  val accumPending   = RegNext(accumIssue, false.B)
  val writeRowPipe   = RegEnable(io.writeRow, accumIssue)
  val saInPipe       = RegEnable(io.saIn, accumIssue)
  val accumWriteData = Wire(Vec(lanes, accType.cloneType))
  for (lane <- 0 until lanes) {
    accumWriteData(lane) := sramReadData(lane) + saInPipe(lane)
  }

  val writeEn   = WireDefault(false.B)
  val writeRow  = WireDefault(io.writeRow)
  val writeData = WireDefault(zeroLine)

  when(io.op === ClassicalAccOp.Store) {
    writeEn   := true.B
    writeData := io.saIn
  }

  when(accumPending) {
    writeEn   := true.B
    writeRow  := writeRowPipe
    writeData := accumWriteData
  }

  assert(
    !(accumPending && io.op === ClassicalAccOp.Store),
    "ClassicalAccumulator has one SRAM write port; do not overlap Accumulate completion with Store"
  )

  for (lane <- 0 until lanes) {
    when(writeEn) {
      banks(lane).write(writeRow, writeData(lane))
    }
  }

  io.saOut := sramOutReg
  io.spOut := sramOutReg
}

class ClassicalMemoryHostIO[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val rdAddr = Input(UInt(addrWidth.W))
  val rdData = Output(Vec(portWidth, dataType))
  val wrEn   = Input(Bool())
  val wrAddr = Input(UInt(addrWidth.W))
  val wrData = Input(Vec(portWidth, dataType))
}

object ClassicalWestSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val Acc = 2.U(2.W); val External = 3.U(2.W)
}

object ClassicalNorthSrc {
  val Zero = 0.U(2.W); val Scratchpad = 1.U(2.W); val Acc = 2.U(2.W); val External = 3.U(2.W)
}

object ClassicalAccInSrc {
  val Zero = 0.U(2.W); val Array = 1.U(2.W); val External = 2.U(2.W)
}

object ClassicalSpWriteSrc {
  val External = 0.U(2.W); val Acc = 1.U(2.W)
}

class ClassicalSystem[T <: Data: Arithmetic](
    N: Int,
    dataType: T,
    accType: T,
    scratchpadDepth: Int,
    accumulatorDepth: Int = 0
) extends Module {
  private val resolvedAccumulatorDepth = if (accumulatorDepth > 0) accumulatorDepth else N
  require(N > 0 && scratchpadDepth >= N && resolvedAccumulatorDepth > 0, "invalid ClassicalSystem dimensions")

  val ev = implicitly[Arithmetic[T]]
  import ev._

  private val spAddrWidth  = SramMacroConfig.scratchpadAddrWidth(N, scratchpadDepth)
  private val accRowIdxW   = math.max(1, log2Ceil(resolvedAccumulatorDepth))
  private val zeroDataLine = 0.U.asTypeOf(Vec(N, dataType.cloneType))
  private val zeroAccLine  = 0.U.asTypeOf(Vec(N, accType.cloneType))

  val io = IO(new Bundle {
    val sp = new ClassicalMemoryHostIO(spAddrWidth, N, dataType)
    val ctrl = Input(new Bundle {
      val spReadEn, spWriteEn     = Bool()
      val spReadAddr, spWriteAddr = UInt(spAddrWidth.W)
      val spWriteSrc              = UInt(2.W)
      val spWriteData             = Vec(N, dataType.cloneType)

      val westSrc      = UInt(2.W)
      val westOp       = Vec(N, ClassicalWestOp())
      val westExternal = Vec(N, dataType.cloneType)

      val northSrc      = UInt(2.W)
      val northExternal = Vec(N, accType.cloneType)

      val accInSrc                = UInt(2.W)
      val accExternal             = Vec(N, accType.cloneType)
      val accOp                   = ClassicalAccOp()
      val accReadRow, accWriteRow = UInt(accRowIdxW.W)
    })

    val spCtrlReadData = Output(Vec(N, dataType.cloneType))
    val saSouth        = Output(Vec(N, accType.cloneType))
    val saSouthValid   = Output(Bool())
    val accSaOut       = Output(Vec(N, accType.cloneType))
    val accSpOut       = Output(Vec(N, accType.cloneType))
  })

  val scratchpad = Scratchpad(N, scratchpadDepth, dataType)
  val systolic   = Module(new SystolicArray(N, dataType, accType))
  val accum      = Module(new Accumulator(accType, N, resolvedAccumulatorDepth))

  val spData      = scratchpad.io.ctrlReadData(0)
  val spDataAsAcc = VecInit(spData.map(_.withWidthOf(accType)))
  val accSaData   = accum.io.saOut
  val accSaAsData = VecInit(accSaData.map(_.withWidthOf(dataType)))
  val accSpAsData = VecInit(accum.io.spOut.map(_.clippedToWidthOf(dataType)))

  // Weight preload is now fully direct-drive: callers must present west-lane
  // vectors in the physical order expected by the mesh.
  val westData = MuxLookup(io.ctrl.westSrc, zeroDataLine)(
    Seq(
      ClassicalWestSrc.Scratchpad -> spData,
      ClassicalWestSrc.Acc        -> accSaAsData,
      ClassicalWestSrc.External   -> io.ctrl.westExternal
    )
  )
  val northData = MuxLookup(io.ctrl.northSrc, zeroAccLine)(
    Seq(
      ClassicalNorthSrc.Scratchpad -> spDataAsAcc,
      ClassicalNorthSrc.Acc        -> accSaData,
      ClassicalNorthSrc.External   -> io.ctrl.northExternal
    )
  )
  val accInData = MuxLookup(io.ctrl.accInSrc, zeroAccLine)(
    Seq(
      ClassicalAccInSrc.Array    -> systolic.io.south,
      ClassicalAccInSrc.External -> io.ctrl.accExternal
    )
  )
  val spWriteData = MuxLookup(io.ctrl.spWriteSrc, io.ctrl.spWriteData)(
    Seq(ClassicalSpWriteSrc.Acc -> accSpAsData)
  )

  scratchpad.io.host.rd(0).addr := io.sp.rdAddr
  scratchpad.io.host.wr(0).en   := io.sp.wrEn
  scratchpad.io.host.wr(0).addr := io.sp.wrAddr
  scratchpad.io.host.wr(0).data := io.sp.wrData
  scratchpad.io.ctrlReadEn(0)   := io.ctrl.spReadEn
  scratchpad.io.ctrlReadAddr(0) := io.ctrl.spReadAddr
  scratchpad.io.ctrlWriteEn     := io.ctrl.spWriteEn
  scratchpad.io.ctrlWriteAddr   := io.ctrl.spWriteAddr
  scratchpad.io.ctrlWriteData   := spWriteData
  io.sp.rdData                  := scratchpad.io.host.rd(0).data

  systolic.io.west := VecInit(Seq.tabulate(N) { row =>
    val token = Wire(new ClassicalWestToken(dataType))
    token.op   := io.ctrl.westOp(row)
    token.data := westData(row)
    token
  })
  systolic.io.north := northData

  accum.io.op       := io.ctrl.accOp
  accum.io.saIn     := accInData
  accum.io.readRow  := io.ctrl.accReadRow
  accum.io.writeRow := io.ctrl.accWriteRow

  io.spCtrlReadData := spData
  io.saSouth        := systolic.io.south
  io.saSouthValid   := systolic.io.southValid
  io.accSaOut       := accum.io.saOut
  io.accSpOut       := accum.io.spOut
}
