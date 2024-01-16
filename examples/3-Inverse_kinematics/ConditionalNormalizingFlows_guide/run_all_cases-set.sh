# Run all the cases
sbatch job_GILBRETH.slurm -n_steps=200000 -num_particles=32 -n_y=64 -lr=1e-3 -realnvp_architecture="[15,100]" -job='a' 'a.out' ####