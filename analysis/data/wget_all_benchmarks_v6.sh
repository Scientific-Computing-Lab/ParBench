#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${1:-downloads}
mkdir -p "$OUTDIR"

echo "Downloading benchmark sources into $OUTDIR"

echo "- BabelStream"
wget -q --show-progress -O "$OUTDIR/BabelStream.zip" "https://codeload.github.com/UoB-HPC/BabelStream/zip/refs/heads/main" || echo "FAILED: BabelStream"

echo "- miniBUDE"
wget -q --show-progress -O "$OUTDIR/miniBUDE.zip" "https://codeload.github.com/UoB-HPC/miniBUDE/zip/refs/heads/main" || echo "FAILED: miniBUDE"

echo "- CloverLeaf"
wget -q --show-progress -O "$OUTDIR/CloverLeaf.zip" "https://codeload.github.com/UoB-HPC/CloverLeaf/zip/refs/heads/main" || echo "FAILED: CloverLeaf"

echo "- LULESH"
wget -q --show-progress -O "$OUTDIR/LULESH.zip" "https://codeload.github.com/LLNL/LULESH/zip/refs/heads/master" || echo "FAILED: LULESH"

echo "- Parallel Research Kernels (PRK)"
wget -q --show-progress -O "$OUTDIR/Parallel_Research_Kernels_PRK.zip" "https://codeload.github.com/ParRes/Kernels/zip/refs/heads/main" || echo "FAILED: Parallel Research Kernels (PRK)"

echo "- TeaLeaf"
wget -q --show-progress -O "$OUTDIR/TeaLeaf.zip" "https://codeload.github.com/UoB-HPC/TeaLeaf/zip/refs/heads/master" || echo "FAILED: TeaLeaf"

echo "- miniMD"
wget -q --show-progress -O "$OUTDIR/miniMD.zip" "https://codeload.github.com/Mantevo/miniMD/zip/refs/heads/master" || echo "FAILED: miniMD"

echo "- HeCBench"
wget -q --show-progress -O "$OUTDIR/HeCBench.zip" "https://codeload.github.com/zjin-lcf/HeCBench/zip/refs/heads/master" || echo "FAILED: HeCBench"

echo "- Kripke"
wget -q --show-progress -O "$OUTDIR/Kripke.zip" "https://codeload.github.com/LLNL/Kripke/zip/refs/heads/develop" || echo "FAILED: Kripke"

echo "- LAMMPS"
wget -q --show-progress -O "$OUTDIR/LAMMPS.zip" "https://codeload.github.com/lammps/lammps/zip/refs/heads/develop" || echo "FAILED: LAMMPS"

echo "- Trilinos"
wget -q --show-progress -O "$OUTDIR/Trilinos.zip" "https://codeload.github.com/trilinos/Trilinos/zip/refs/heads/master" || echo "FAILED: Trilinos"

echo "- miniWeather"
wget -q --show-progress -O "$OUTDIR/miniWeather.zip" "https://codeload.github.com/mrnorman/miniWeather/zip/refs/heads/main" || echo "FAILED: miniWeather"

echo "- AMReX"
wget -q --show-progress -O "$OUTDIR/AMReX.zip" "https://codeload.github.com/AMReX-Codes/amrex/zip/refs/heads/development" || echo "FAILED: AMReX"

echo "- MFEM"
wget -q --show-progress -O "$OUTDIR/MFEM.zip" "https://codeload.github.com/mfem/mfem/zip/refs/heads/master" || echo "FAILED: MFEM"

echo "- Neutral"
wget -q --show-progress -O "$OUTDIR/Neutral.zip" "https://codeload.github.com/UoB-HPC/Neutral/zip/refs/heads/master" || echo "FAILED: Neutral"

echo "- RAJA Performance Suite (RAJAPerf)"
wget -q --show-progress -O "$OUTDIR/RAJA_Performance_Suite_RAJAPerf.zip" "https://codeload.github.com/LLNL/RAJAPerf/zip/refs/heads/develop" || echo "FAILED: RAJA Performance Suite (RAJAPerf)"

echo "- SHOC"
wget -q --show-progress -O "$OUTDIR/SHOC.zip" "https://codeload.github.com/vetter/shoc/zip/refs/heads/master" || echo "FAILED: SHOC"

echo "- ExaMiniMD"
wget -q --show-progress -O "$OUTDIR/ExaMiniMD.zip" "https://codeload.github.com/ECP-copa/ExaMiniMD/zip/refs/heads/master" || echo "FAILED: ExaMiniMD"

echo "- GROMACS"
wget -q --show-progress -O "$OUTDIR/GROMACS.tar.gz" "https://gitlab.com/gromacs/gromacs/-/archive/main/gromacs-main.tar.gz" || echo "FAILED: GROMACS"

echo "- Kokkos Kernels"
wget -q --show-progress -O "$OUTDIR/Kokkos_Kernels.zip" "https://codeload.github.com/kokkos/kokkos-kernels/zip/refs/heads/develop" || echo "FAILED: Kokkos Kernels"

echo "- PENNANT"
wget -q --show-progress -O "$OUTDIR/PENNANT.zip" "https://codeload.github.com/lanl/PENNANT/zip/refs/heads/master" || echo "FAILED: PENNANT"

echo "- Parboil"
wget -q --show-progress -O "$OUTDIR/Parboil.zip" "https://codeload.github.com/utcs-scea/parboil/zip/refs/heads/master" || echo "FAILED: Parboil"

echo "- Rodinia"
wget -q --show-progress -O "$OUTDIR/Rodinia.zip" "https://codeload.github.com/yuhc/gpu-rodinia/zip/refs/heads/master" || echo "FAILED: Rodinia"

echo "- SW4lite"
wget -q --show-progress -O "$OUTDIR/SW4lite.zip" "https://codeload.github.com/geodynamics/sw4lite/zip/refs/heads/master" || echo "FAILED: SW4lite"

echo "- CoMD"
wget -q --show-progress -O "$OUTDIR/CoMD.tar.gz" "https://github.com/exmatex/CoMD/archive/v1.1.tar.gz" || echo "FAILED: CoMD"

echo "- HPC Challenge (HPCC)"
wget -q --show-progress -O "$OUTDIR/HPC_Challenge_HPCC.zip" "https://codeload.github.com/icl-utk-edu/hpcc/zip/refs/heads/master" || echo "FAILED: HPC Challenge (HPCC)"

echo "- OSU Micro-Benchmarks (OMB)"
wget -q --show-progress -O "$OUTDIR/OSU_Micro_Benchmarks_OMB.tar.gz" "https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-8.0b2.tar.gz" || echo "FAILED: OSU Micro-Benchmarks (OMB)"

echo "- OSU PGAS Benchmarks (OMB extensions)"
wget -q --show-progress -O "$OUTDIR/OSU_PGAS_Benchmarks_OMB_extensions.tar.gz" "https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-8.0b2.tar.gz" || echo "FAILED: OSU PGAS Benchmarks (OMB extensions)"

echo "- OpenDwarfs"
wget -q --show-progress -O "$OUTDIR/OpenDwarfs.zip" "https://codeload.github.com/vtsynergy/opendwarfs/zip/refs/heads/master" || echo "FAILED: OpenDwarfs"

echo "- PARSEC"
wget -q --show-progress -O "$OUTDIR/PARSEC.zip" "https://codeload.github.com/bamos/parsec-benchmark/zip/refs/heads/master" || echo "FAILED: PARSEC"

echo "- RAJAProxies (CoMD/Kripke/LULESH)"
wget -q --show-progress -O "$OUTDIR/RAJAProxies_CoMD_Kripke_LULESH.zip" "https://codeload.github.com/LLNL/RAJAProxies/zip/refs/heads/main" || echo "FAILED: RAJAProxies (CoMD/Kripke/LULESH)"

echo "- STAMP"
wget -q --show-progress -O "$OUTDIR/STAMP.zip" "https://codeload.github.com/ddcc/stamp/zip/refs/heads/master" || echo "FAILED: STAMP"

echo "- miniAMR"
wget -q --show-progress -O "$OUTDIR/miniAMR.zip" "https://codeload.github.com/Mantevo/miniAMR/zip/refs/heads/master" || echo "FAILED: miniAMR"

echo "- miniFE"
wget -q --show-progress -O "$OUTDIR/miniFE.zip" "https://codeload.github.com/Mantevo/miniFE/zip/refs/heads/master" || echo "FAILED: miniFE"

echo "- HPCCG"
wget -q --show-progress -O "$OUTDIR/HPCCG.zip" "https://codeload.github.com/Mantevo/HPCCG/zip/refs/heads/master" || echo "FAILED: HPCCG"

echo "- HPCG"
wget -q --show-progress -O "$OUTDIR/HPCG.zip" "https://codeload.github.com/hpcg-benchmark/hpcg/zip/refs/heads/master" || echo "FAILED: HPCG"

echo "- NAS Parallel Benchmarks (NPB)"
wget -q --show-progress -O "$OUTDIR/NAS_Parallel_Benchmarks_NPB.tar.gz" "https://www.nas.nasa.gov/assets/npb/NPB3.4.3.tar.gz" || echo "FAILED: NAS Parallel Benchmarks (NPB)"

echo "- HPL (High-Performance Linpack)"
wget -q --show-progress -O "$OUTDIR/HPL_High_Performance_Linpack.tar.gz" "https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz" || echo "FAILED: HPL (High-Performance Linpack)"

echo "- STREAM"
wget -q --show-progress -O "$OUTDIR/STREAM.zip" "https://codeload.github.com/jeffhammond/STREAM/zip/refs/heads/master" || echo "FAILED: STREAM"

echo "- miniGhost"
wget -q --show-progress -O "$OUTDIR/miniGhost.zip" "https://codeload.github.com/Mantevo/miniGhost/zip/refs/heads/master" || echo "FAILED: miniGhost"
