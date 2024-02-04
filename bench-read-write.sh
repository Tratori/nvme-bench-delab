node_config=("nx01" "gx05" "nx02" "gx03" "gx04" "gx01_raid1")
declare -A mounts
declare -A nodenames
declare -A files
declare -A partitions
declare -A ssds
#mounts["nx01"]="/mnt/nvme3:/mnt/nvme3,/mnt/nvme2:/mnt/nvme2"
#files["nx01"]="/mnt/nvme3/file_nvme_bench;/mnt/nvme2/file_nvme_bench"

mounts["nx01"]="/mnt/nvme2:/mnt/nvme2"
files["nx01"]="/mnt/nvme2/file_nvme_bench"
partitions["nx01"]="alchemy"
ssds["nx01"]="INTEL_SSDPE2KE016T8"
nodenames["nx01"]="nx01"

mounts["gx05"]="/scratch:/scratch"
files["gx05"]="/scratch/file_nvme_bench"
partitions["gx05"]="sorcery"
ssds["gx05"]="Dell_Ent_NVMe_v2_AGN_RI_U.2_1.92TB"
nodenames["gx05"]="gx05"

mounts["nx02"]="/mnt/raid/:/mnt/raid/"
files["nx02"]="/mnt/raid/file_nvme_bench"
partitions["nx02"]="alchemy"
ssds["nx02"]="4_SSD_Raid0"
nodenames["nx02"]="nx02"

mounts["gx03"]="/scratch/:/scratch/"
files["gx03"]="/scratch/file_nvme_bench"
partitions["gx03"]="sorcery"
ssds["gx03"]="1_SSD"
nodenames["gx03"]="gx03"

mounts["gx04"]="/scratch/:/scratch/"
files["gx04"]="/scratch/file_nvme_bench"
partitions["gx04"]="sorcery"
ssds["gx04"]="1_SSD"
nodenames["gx04"]="gx04"

mounts["gx01_raid0"]="/mnt/userspace/scratch:/mnt/userspace/scratch"
files["gx01_raid0"]="/mnt/userspace/scratch/file_nvme_bench"
partitions["gx01_raid0"]="sorcery"
ssds["gx01_raid0"]="4_SSDs"
nodenames["gx01_raid0"]="gx01"

mounts["gx01_raid1"]="/scratch/:/scratch/"
files["gx01_raid1"]="/scratch/file_nvme_bench"
partitions["gx01_raid1"]="sorcery"
ssds["gx01_raid1"]="2_SSDs"
nodenames["gx01_raid1"]="gx01"

mounts["gx01_full"]="/scratch/:/scratch/,/mnt/userspace/scratch:/mnt/userspace/scratch"
files["gx01_full"]="/scratch/file_nvme_bench;/mnt/userspace/scratch/file_nvme_bench"
partitions["gx01_full"]="sorcery"
ssds["gx01_full"]="6_SSDs"
nodenames["gx01_full"]="gx01"

mounts["cx17"]="/tmp:/tmp"
files["cx17"]="/tmp/file_nvme_bench"
partitions["cx17"]="magic"
ssds["cx17"]="MZXL5800HBHQ-000H3"
nodenames["cx17"]="cx17"

mounts["ca06"]="/tmp:/tmp"
files["ca06"]="/tmp/file_nvme_bench"
partitions["ca06"]="magic"
ssds["ca06"]="Micron_7300_MTFDHBA400TDG"
nodenames["ca06"]="ca06"

mounts["cp01"]="/scratch:/scratch"
files["cp01"]="/scratch/file_nvme_bench"
partitions["cp01"]="alchemy"
ssds["cp01"]="1.6TB_NVMe_Gen4_U.2_SSD"
nodenames["cp01"]="cp01"

for node_conf in ${node_config[@]}; do
    node=${nodenames[$node_conf]}
    RESULT_FILE="leanstore/results/$node_conf"

    node_files=("${files[$node_conf]}")

    mkdir -p $(pwd)/leanstore/results/$node_conf

    echo "submitting task for config ${node_conf}"
    srun -A rabl --partition ${partitions[$node_conf]} -w $node -c 32 --mem-per-cpu 1024 \
      --time=12:00:00 --container-image=/hpi/fs00/share/fg-rabl/dpmh23_nvme/leanstore_all_dep.sqsh \
      --container-mounts=$(pwd)/leanstore:/leanstore,${mounts[$node_conf]},$(pwd)/nvme-bench-delab:/nvme-bench-delab  \
      python3 /nvme-bench-delab/bench.py \
        $node_files /leanstore/build/frontend/iob /leanstore/results/$node_conf/benchmark ${ssds[$node_conf]} /nvme-bench-delab/workloads/mixed_read_write.yaml mixed_read_write &
done
