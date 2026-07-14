# RTL provenance for the paper mesh sweep

The retained synthesis inputs are the emitted SystemVerilog under `paper-mesh-snapshot/`. Their file hashes are recorded in `RTL_SNAPSHOT.sha256`, and `RTL_MANIFEST.csv` maps each of the 112 selected synthesis jobs to one snapshot directory.

The main mesh upload archives were created on 2026-04-06 at approximately 02:53 Asia/Shanghai, and the selected synth-site projects were created immediately afterward at 2026-04-05 18:53 UTC. The corrected FSA FP8 archive and job were created on April 7. `RTL_MANIFEST.csv` records whether each directory can still be checked byte-for-byte against its retained upload ZIP. Most can; four FSA FP16 ZIPs were overwritten by the later FP8 retry, so their pre-upload emitted directories are retained but cannot be re-proven against the original ZIP.

There is no honest single Git commit for all of these emitted files. The last architecture commits before upload were `c0dff6a6d3afe5effa8ffedf75b774dfcfe06cb9` and `dfb9839ab71c7806f9c1d3f24732cf0b9b7c2741`, but the `GenerateMeshes` driver was still uncommitted at upload time and was committed later as `a561d4bbeeac53f43ba17b113e4a19f5841942a3`. Regenerating a representative classical N=4 FP16 case from the later committed source did not match the archived top-level `SystolicArray.sv` (although `PE.sv` did match), so the bundle does not mislabel a nearby commit as exact.

For artifact evaluation, use `paper-mesh-snapshot/` together with the per-job provenance class in `RTL_MANIFEST.csv`. The Chisel source in this directory remains useful for regeneration and inspection, but newly generated RTL is a current implementation check, not a claim of byte-identical reconstruction of every April synthesis input.
