package pinn.common

import chisel3._
import chisel3.util._

class BankedSramRead[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val addr = Input(UInt(addrWidth.W))
  val data = Output(Vec(portWidth, dataType))
}

class BankedSramWrite[T <: Data](addrWidth: Int, portWidth: Int, dataType: T) extends Bundle {
  val en   = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val data = Input(Vec(portWidth, dataType))
}

class BankedSramIO[T <: Data](addrWidth: Int, portWidth: Int, rdPort: Int, wrPort: Int, dataType: T) extends Bundle {
  val rd = Vec(rdPort, new BankedSramRead(addrWidth, portWidth, dataType))
  val wr = Vec(wrPort, new BankedSramWrite(addrWidth, portWidth, dataType))
}

object SramMacroConfig {
  val ScratchpadBankGroups  = 4
  val AccumulatorBankGroups = 4

  def scratchpadBankDepth(requestedDepth: Int): Int = {
    require(requestedDepth > 0, s"scratchpad depth must be positive, got $requestedDepth")
    require(requestedDepth <= 2048, s"scratchpad depth must fit a 1024/2048 macro, got $requestedDepth")
    if (requestedDepth <= 1024) 1024 else 2048
  }

  def accumulatorBankDepth(saSize: Int): Int = {
    require(saSize > 0, s"SA size must be positive, got $saSize")
    math.max(32, saSize)
  }

  def scratchpadAddrWidth(saSize: Int, requestedDepth: Int): Int =
    log2Ceil(ScratchpadBankGroups * saSize * scratchpadBankDepth(requestedDepth))
}

class GroupedRowRead[T <: Data](addrWidth: Int, rowWidth: Int, dataType: T) extends Bundle {
  val en   = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val data = Output(Vec(rowWidth, dataType))
}

class GroupedRowWrite[T <: Data](addrWidth: Int, rowWidth: Int, dataType: T) extends Bundle {
  val en   = Input(Bool())
  val addr = Input(UInt(addrWidth.W))
  val data = Input(Vec(rowWidth, dataType))
}

class GroupedRowMemIO[T <: Data](addrWidth: Int, rowWidth: Int, rdPorts: Int, dataType: T) extends Bundle {
  val rd = Vec(rdPorts, new GroupedRowRead(addrWidth, rowWidth, dataType))
  val wr = new GroupedRowWrite(addrWidth, rowWidth, dataType)
}

class GroupedRowMem[T <: Data](
    visibleGroupCount: Int,
    physicalGroupCount: Int,
    rowWidth: Int,
    logicalBankDepth: Int,
    physicalBankDepth: Int,
    rdPorts: Int,
    dataType: T
) extends Module {
  require(visibleGroupCount > 0, "GroupedRowMem must expose at least one logical bank group")
  require(physicalGroupCount >= visibleGroupCount, "physical bank groups must cover logical bank groups")
  require(rowWidth > 0, "GroupedRowMem rowWidth must be positive")
  require(logicalBankDepth > 0, "GroupedRowMem logical bankDepth must be positive")
  require(physicalBankDepth >= logicalBankDepth, "physical bankDepth must cover logical bankDepth")
  require(rdPorts > 0, "GroupedRowMem must expose at least one read port")

  private val addrWidth     = log2Ceil(visibleGroupCount * rowWidth * logicalBankDepth)
  private val groupIdxWidth = math.max(1, log2Ceil(physicalGroupCount))
  private val rowIdxWidth   = math.max(1, log2Ceil(physicalBankDepth))
  private val rdIdxWidth    = math.max(1, log2Ceil(rdPorts))
  private val zeroLine      = 0.U.asTypeOf(Vec(rowWidth, dataType.cloneType))
  private val banks         = Seq.fill(physicalGroupCount, rowWidth)(SyncReadMem(physicalBankDepth, dataType.cloneType))

  val io = IO(new GroupedRowMemIO(addrWidth, rowWidth, rdPorts, dataType))

  private def decode(addr: UInt, active: Bool, opName: String): (UInt, UInt) = {
    when(active) {
      assert(
        addr < (visibleGroupCount * rowWidth * logicalBankDepth).U,
        s"$opName address must stay within the grouped SRAM space"
      )
      assert(addr % rowWidth.U === 0.U, s"$opName address must be row-aligned")
    }

    val lineIdx = addr / rowWidth.U
    val group   = (lineIdx / logicalBankDepth.U)(groupIdxWidth - 1, 0)
    val row     = (lineIdx % logicalBankDepth.U)(rowIdxWidth - 1, 0)
    (group, row)
  }

  private val decodedReads = Seq.tabulate(rdPorts) { rdIdx =>
    val (group, row) = decode(io.rd(rdIdx).addr, io.rd(rdIdx).en, s"grouped SRAM read $rdIdx")
    (group, row)
  }

  private val groupReadData = Seq.tabulate(physicalGroupCount) { groupIdx =>
    val hitVec = VecInit(
      Seq.tabulate(rdPorts) { rdIdx =>
        io.rd(rdIdx).en && decodedReads(rdIdx)._1 === groupIdx.U
      }
    )
    val winnerValid = hitVec.asUInt.orR
    val winnerIdx   = PriorityEncoder(hitVec)
    val winnerIdxReg = RegEnable(winnerIdx, winnerValid)
    val winnerValidReg = RegNext(winnerValid, false.B)
    val winnerRow = Mux1H(
      Seq.tabulate(rdPorts) { rdIdx =>
        hitVec(rdIdx) -> decodedReads(rdIdx)._2
      }
    )
    val readData = VecInit(
      Seq.tabulate(rowWidth) { lane =>
        banks(groupIdx)(lane).read(winnerRow, winnerValid)
      }
    )
    (readData, winnerIdxReg, winnerValidReg)
  }

  for (rdIdx <- 0 until rdPorts) {
    val rdData = WireDefault(zeroLine)
    for (groupIdx <- 0 until physicalGroupCount) {
      val (readData, winnerIdxReg, winnerValidReg) = groupReadData(groupIdx)
      when(winnerValidReg && winnerIdxReg === rdIdx.U(rdIdxWidth.W)) {
        rdData := readData
      }
    }
    io.rd(rdIdx).data := rdData
  }

  private val (wrGroup, wrRow) = decode(io.wr.addr, io.wr.en, "grouped SRAM write")
  for (groupIdx <- 0 until physicalGroupCount) {
    when(io.wr.en && wrGroup === groupIdx.U) {
      for (lane <- 0 until rowWidth) {
        banks(groupIdx)(lane).write(wrRow, io.wr.data(lane))
      }
    }
  }
}

class LaneBankRead[T <: Data](addrWidth: Int, laneCount: Int, dataType: T) extends Bundle {
  val en   = Input(Vec(laneCount, Bool()))
  val addr = Input(Vec(laneCount, UInt(addrWidth.W)))
  val data = Output(Vec(laneCount, dataType))
}

class LaneBankWrite[T <: Data](addrWidth: Int, laneCount: Int, dataType: T) extends Bundle {
  val en   = Input(Vec(laneCount, Bool()))
  val addr = Input(Vec(laneCount, UInt(addrWidth.W)))
  val data = Input(Vec(laneCount, dataType))
}

class LaneBankMemIO[T <: Data](addrWidth: Int, laneCount: Int, rdPorts: Int, dataType: T) extends Bundle {
  val rd = Vec(rdPorts, new LaneBankRead(addrWidth, laneCount, dataType))
  val wr = new LaneBankWrite(addrWidth, laneCount, dataType)
}

class LaneBankMem[T <: Data](laneCount: Int, logicalDepth: Int, physicalDepth: Int, rdPorts: Int, dataType: T)
    extends Module {
  require(laneCount > 0, "LaneBankMem laneCount must be positive")
  require(logicalDepth > 0, "LaneBankMem logicalDepth must be positive")
  require(physicalDepth >= logicalDepth, "LaneBankMem physicalDepth must cover logicalDepth")
  require(rdPorts > 0, "LaneBankMem must expose at least one read port")

  private val addrWidth = math.max(1, log2Ceil(physicalDepth))
  private val rdIdxWidth = math.max(1, log2Ceil(rdPorts))
  private val banks     = Seq.fill(laneCount)(SyncReadMem(physicalDepth, dataType.cloneType))
  private val zeroElem  = 0.U.asTypeOf(dataType.cloneType)

  val io = IO(new LaneBankMemIO(addrWidth, laneCount, rdPorts, dataType))

  for (lane <- 0 until laneCount) {
    val hitVec = VecInit(Seq.tabulate(rdPorts)(rdIdx => io.rd(rdIdx).en(lane)))
    for (rdIdx <- 0 until rdPorts) {
      when(io.rd(rdIdx).en(lane)) {
        assert(io.rd(rdIdx).addr(lane) < logicalDepth.U, s"lane-bank read $rdIdx must stay within logical depth")
      }
    }

    val winnerValid    = hitVec.asUInt.orR
    val winnerIdx      = PriorityEncoder(hitVec)
    val winnerIdxReg   = RegEnable(winnerIdx, winnerValid)
    val winnerValidReg = RegNext(winnerValid, false.B)
    val winnerAddr = Mux1H(
      Seq.tabulate(rdPorts) { rdIdx =>
        hitVec(rdIdx) -> io.rd(rdIdx).addr(lane)
      }
    )
    val readData = banks(lane).read(winnerAddr, winnerValid)

    for (rdIdx <- 0 until rdPorts) {
      io.rd(rdIdx).data(lane) := Mux(
        winnerValidReg && winnerIdxReg === rdIdx.U(rdIdxWidth.W),
        readData,
        zeroElem
      )
    }
  }

  for (lane <- 0 until laneCount) {
    when(io.wr.en(lane)) {
      assert(io.wr.addr(lane) < logicalDepth.U, "lane-bank write must stay within logical depth")
      banks(lane).write(io.wr.addr(lane), io.wr.data(lane))
    }
  }
}

class ScratchpadIO[T <: Data](addrWidth: Int, portWidth: Int, ctrlReadPorts: Int, dataType: T) extends Bundle {
  val host = new BankedSramIO(addrWidth, portWidth, 1, 1, dataType)

  val ctrlReadAddr = Input(Vec(ctrlReadPorts, UInt(addrWidth.W)))
  val ctrlReadEn   = Input(Vec(ctrlReadPorts, Bool()))
  val ctrlReadData = Output(Vec(ctrlReadPorts, Vec(portWidth, dataType)))

  val ctrlWriteEn   = Input(Bool())
  val ctrlWriteAddr = Input(UInt(addrWidth.W))
  val ctrlWriteData = Input(Vec(portWidth, dataType))
}

class Scratchpad[T <: Data: Arithmetic](N: Int, depth: Int, ctrlReadPorts: Int, dataType: T) extends Module {
  require(ctrlReadPorts > 0, "Scratchpad must expose at least one controller read port")
  require(ctrlReadPorts <= 2, s"Scratchpad exposes up to 2 logical controller read ports, got $ctrlReadPorts")

  private val bankDepth = SramMacroConfig.scratchpadBankDepth(depth)
  private val addrWidth = SramMacroConfig.scratchpadAddrWidth(N, depth)
  private val zeroLine  = 0.U.asTypeOf(Vec(N, dataType.cloneType))

  val io = IO(new ScratchpadIO(addrWidth, N, ctrlReadPorts, dataType))

  val sram = Module(
    new GroupedRowMem(
      SramMacroConfig.ScratchpadBankGroups,
      SramMacroConfig.ScratchpadBankGroups,
      N,
      bankDepth,
      bankDepth,
      1,
      dataType
    )
  )

  assert(PopCount(io.ctrlReadEn) <= 1.U, "Scratchpad 1R path allows at most one logical controller read per cycle")
  assert(!(io.host.wr(0).en && io.ctrlWriteEn), "host and controller cannot write scratchpad together")

  private val ctrlReadSel = WireDefault(0.U(2.W))
  when(io.ctrlReadEn(0)) {
    ctrlReadSel := 1.U
  }
  if (ctrlReadPorts > 1) {
    when(io.ctrlReadEn(1)) {
      ctrlReadSel := 2.U
    }
  }

  private val sharedReadAddr = WireDefault(io.host.rd(0).addr)
  when(ctrlReadSel === 1.U) {
    sharedReadAddr := io.ctrlReadAddr(0)
  }
  if (ctrlReadPorts > 1) {
    when(ctrlReadSel === 2.U) {
      sharedReadAddr := io.ctrlReadAddr(1)
    }
  }
  private val sharedReadSelReg = RegNext(ctrlReadSel, 0.U)

  sram.io.rd(0).en   := true.B
  sram.io.rd(0).addr := sharedReadAddr

  private val hostReadData = WireDefault(zeroLine)
  private val ctrlReadData = WireDefault(VecInit(Seq.fill(ctrlReadPorts)(zeroLine)))

  when(sharedReadSelReg === 0.U) {
    hostReadData := sram.io.rd(0).data
  }.elsewhen(sharedReadSelReg === 1.U) {
    ctrlReadData(0) := sram.io.rd(0).data
  }
  if (ctrlReadPorts > 1) {
    when(sharedReadSelReg === 2.U) {
      ctrlReadData(1) := sram.io.rd(0).data
    }
  }

  private val writeEn   = io.host.wr(0).en || io.ctrlWriteEn
  private val writeAddr = Mux(io.ctrlWriteEn, io.ctrlWriteAddr, io.host.wr(0).addr)

  sram.io.wr.en   := writeEn
  sram.io.wr.addr := writeAddr
  sram.io.wr.data := Mux(io.ctrlWriteEn, io.ctrlWriteData, io.host.wr(0).data)

  io.host.rd(0).data := hostReadData
  io.ctrlReadData    := ctrlReadData
}

object Scratchpad {
  def apply[T <: Data: Arithmetic](N: Int, depth: Int, dataType: T): Scratchpad[T] =
    Module(new Scratchpad(N, depth, 1, dataType))

  def apply[T <: Data: Arithmetic](N: Int, depth: Int, ctrlReadPorts: Int, dataType: T): Scratchpad[T] =
    Module(new Scratchpad(N, depth, ctrlReadPorts, dataType))
}
