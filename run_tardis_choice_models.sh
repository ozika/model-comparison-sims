PATH_BASE="${HOME}/projects/model-comparison-sims"
# Give message to user
#echo "${SUB_LABEL}: ${MODEL}"
N_CPUS=1
# maximum number of threads per process:
N_THREADS=1
# memory demand in *GB*
MEM_MB=1
# data directory
#PATH_DATA="${PATH_BASE}/data/temp"

#models=("mm1_RW_outcome" "mm1fam1_RW_outcome" "mm2fam1_RW_phase" "mm2_RW_phase" "mm3fam1_RW_outcome_phase" "mm3_RW_outcome_phase" "mm3_RW_outcome_phase_ver2" "mm4fam1_RW_lapse" "mm4_RW_lapse" "mm5fam1_PH_hybrid"  "mm5v3fam1_PH_hybrid" "mm5_PH_hybrid_outcome" "mm5v2_PH_hybrid_outcome")
#flag="13_models"
#models=("mm1_nh_RW_outcome" "mm2_nh_RW_phase" "mm3_nh_RW_outcome_phase" "mm4_nh_RW_lapse" "mm4_nh_RW_lapse_outcome" "mm5_nh_PH_hybrid" "mm5_nh_PH_hybrid_outcome")
#models=("mr1_RW_outcome_visit_group_hyperprior","mr1_RW_outcome_visit_hyperprior")
#flag="varying_prior_models"

#flag="grpr_models"
#models=("mm1_grpr_RW_outcome" "mm2_grpr_RW_phase" "mm3_grpr_RW_outcome_phase" "mm4_grpr_RW_lapse" "mm5_grpr_PH_hybrid" "mm5_grpr_PH_hybrid_outcome")

#flag="vispr_models"
#models=("mm1_vispr_RW_outcome" "mm2_vispr_RW_phase" "mm3_vispr_RW_outcome_phase" "mm4_vispr_RW_lapse" "mm5_vispr_PH_hybrid" "mm5_vispr_PH_hybrid_outcome")

#flag="grvispr_models"
#algos=("mm1_nh_RW_outcome" "mm2_nh_RW_phase" "mm3_nh_RW_outcome_phase" "mm4_nh_RW_lapse" "mm4_nh_RW_lapse_outcome" "mm5_nh_PH_hybrid" "mm5_nh_PH_hybrid_outcome" "mm5_nh_PH_hybrid_kappa" "mm5_nh_PH_hybrid_outcome_kappa")

#flag="nh_models"
#models=("mm5_nh_PH_hybrid_kappa" "mm5_nh_PH_hybrid_outcome_kappa")
algorithms=("COBYLA" "Nelder-Mead" "CG" "BFGS")
noise_lvl=(0.1 0.2 0.4 0.7) 
val_noise_lvl=(5 10 20 40) 
cutoff=(60 120 150)
niter=50
cond="choice_models_C"

for algo in "${algorithms[@]}"; do
  for cn in "${noise_lvl[@]}"; do
    for vn in "${val_noise_lvl[@]}"; do
        for c in "${cutoff[@]}"; do
        JOB_NAME="mc_i${niter}_n${n}_c${c}_${algo}"
        # Create job file
        echo "#!/bin/bash" > job.slurm
        # name of the job
        echo "#SBATCH --job-name ${JOB_NAME}" >> job.slurm
        # set the expected maximum running time for the job:
        echo "#SBATCH --time 99:59:00" >> job.slurm
        # determine how much RAM your operation needs:
        echo "#SBATCH --mem ${MEM_MB}GB" >> job.slurm
        # determine number of CPUs
        echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job.slurm
        # write to log folder
        #echo "#SBATCH --output ${PATH_LOG}/slurm-${JOB_NAME}.%j.out" >> job.slurm
        echo "#SBATCH --output /home/mpib/zika/logs/slurm_${JOB_NAME}_%j.out" >> job.slurm
        echo "#SBATCH --partition long" >> job.slurm
        echo "mkdir ${PATH_BASE}/data/${cond}/" >> job.slurm
        # Load R module
        echo "module load conda" >> job.slurm
        echo "conda activate mcenv" >> job.slurm
        echo "pip install numpy scipy pandas matplotlib groo-ozika==0.1.1" >> job.slurm
        echo "python ${PATH_BASE}/sim_choices_mc.py \
        -a ${algo} -x ${cn} -n ${vn} -c ${c} -i ${niter} -d ${cond}" >> job.slurm

        # submit job to cluster queue and remove it to avoid confusion:
        #cat job.slurm
        sbatch job.slurm
        rm -f job.slurm
        sleep 4
        done
    done
  done
done
