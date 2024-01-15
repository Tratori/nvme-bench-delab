nodes=("nx01" "gx05")
declare -A mounts
declare -A files
declare -A partitions
declare -A ssds
#mounts["nx01"]="/mnt/nvme3:/mnt/nvme3,/mnt/nvme2:/mnt/nvme2"
#files["nx01"]="/mnt/nvme3/file_nvme_bench;/mnt/nvme2/file_nvme_bench"

mounts["nx01"]="/mnt/nvme2:/mnt/nvme2"
files["nx01"]="/mnt/nvme2/file_nvme_bench"
partitions["nx01"]="alchemy"
ssds["nx01"]="INTEL_SSDPE2KE016T8"

mounts["gx05"]="/scratch:/scratch"
files["gx05"]="/scratch/file_nvme_bench"
partitions["gx05"]="sorcery"
ssds["gx05"]="Dell_Ent_NVMe_v2_AGN_RI_U.2_1.92TB"

mounts["nx02_raid0"]="/mnt/raid/:/mnt/raid/"
files["nx02_raid0"]="/mnt/raid/file_nvme_bench"
partitions["nx02_raid0"]="sorcery"
ssds["nx02_raid0"]="4_SSD_Raid0"

mounts["gx03"]="/scratch/:/scratch/"
files["gx03"]="/scratch/file_nvme_bench"
partitions["gx03"]="sorcery"
ssds["gx03"]="1_SSD"

mounts["gx04"]="/scratch/:/scratch/"
files["gx04"]="/scratch/file_nvme_bench"
partitions["gx04"]="sorcery"
ssds["gx04"]="1_SSD"

mounts["gx01_raid0"]="/mnt/userspace:/mnt/userspace"
files["gx01_raid0"]="/mnt/userspace/file_nvme_bench"
partitions["gx01_raid0"]="sorcery"
ssds["gx01_raid0"]="4_SSDs"

mounts["gx01_raid1"]="/tmp/:/tmp/"
files["gx01_raid1"]="/tmp/file_nvme_bench"
partitions["gx01_raid1"]="sorcery"
ssds["gx01_raid1"]="2_SSDs"

mounts["gx01_full"]="/tmp/:/tmp/,/mnt/userspace:/mnt/userspace"
files["gx01_full"]="/tmp/file_nvme_bench;/tmp/:/tmp/file_nvme_bench"
partitions["gx01_full"]="sorcery"
ssds["gx01_full"]="6_SSDs"

mounts["cx17"]="/tmp:/tmp"
files["cx17"]="/tmp/file_nvme_bench"
partitions["cx17"]="magic"
ssds["cx17"]="MZXL5800HBHQ-000H3"

mounts["ca06"]="/tmp:/tmp"
files["ca06"]="/tmp/file_nvme_bench"
partitions["ca06"]="magic"
ssds["ca06"]="Micron_7300_MTFDHBA400TDG"

mounts["cp01"]="/scratch:/scratch"
files["cp01"]="/scratch/file_nvme_bench"
partitions["cp01"]="alchemy"
ssds["cp01"]="1.6TB_NVMe_Gen4_U.2_SSD"

for node in ${nodes[@]}; do

    RESULT_FILE="leanstore/results/$node"

    node_files=("${files[$node]}")

    mkdir -p $(pwd)/leanstore/results/$node

    echo "submitting task for node ${node}"
    srun -A rabl --partition ${partitions[$node]} -w $node -c 32 --mem-per-cpu 1024 \
      --container-image=$(pwd)/leanstore_all_dep.sqsh \
      --container-mounts=$(pwd)/leanstore:/leanstore,${mounts[$node]} \
      python3 /leanstore/bench.py \
        $node_files /leanstore/build/frontend/iob /leanstore/results/$node/benchmark ${ssds[$node]} &
done
