# Environment for the SLURM job scripts. Copy it into place once:
#
#   cp scripts/env.example.sh $HOME/tot/env.sh
#
# scripts/slurm_experiment.sh (and, through it, slurm_experiment_batched.sh)
# and scripts/slurm_benchmark.sh `source $HOME/tot/env.sh` at the top of every
# job. Without it every task fails with "dotnet: command not found".
#
# WHY A SOURCED FILE AND NOT ~/.bashrc. SLURM batch shells are non-interactive,
# and a non-interactive bash does not read .bashrc -- so a PATH set there is
# present on the login node, where everything appears to work, and absent in
# every job. Sourcing an explicit file makes the job's environment the same on
# the login node and on a compute node.
#
# This assumes the per-user SDK install described in REPRODUCE.md
# (Prerequisites): CLAIX has no .NET module, so the SDK was installed with
# Microsoft's dotnet-install.sh into $HOME/dotnet. Edit DOTNET_ROOT if yours
# went elsewhere.

export DOTNET_ROOT="$HOME/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
