# Environment for the SLURM job scripts. Copy it into place once:
#
#   cp scripts/env.example.sh $HOME/tot/env.sh
#
# scripts/slurm_experiment.sh (and through it slurm_experiment_batched.sh)
# sources $HOME/tot/env.sh at the top of every job. Without it every task fails
# with "dotnet: command not found".
#
# A sourced file rather than ~/.bashrc: SLURM batch shells are non-interactive and
# do not read .bashrc, so a PATH set there exists on the login node but not in
# jobs.
#
# Assumes the per-user SDK install in REPRODUCE.md (Prerequisites): CLAIX has no
# .NET module, so the SDK was installed with Microsoft's dotnet-install.sh into
# $HOME/dotnet. Edit DOTNET_ROOT if yours is elsewhere.

export DOTNET_ROOT="$HOME/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
