import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Write script to process cmds on BMRC cluster", add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# input arguments
parser.add_argument('cmdsTxt', type=str, default='process-cmds.txt',
                    help="Text file with one command per line.")
parser.add_argument('-l', '--logdir', type=str, default='./logs', metavar='path',
                    help="Directory path to store log output")

# optional arguments
env_group = parser.add_mutually_exclusive_group()
env_group.add_argument('--venv', type=str, default=None, metavar='path',
                       help="""If you wish to load a pip virtual environment, path to pip
                       virtual environment  without -ivybridge or -skylake suffix. 
                       See BMRC help page for more info on working with pip virtual 
                       environments on BMRC""")
env_group.add_argument('--conda', type=str, default=None, metavar='env',
                       help="""If you wish to activate an Anaconda environment, name of the environment. 
                       See BMRC help page for more info on working with Anaconda environments on BMRC""")
parser.add_argument('--conda-module', type=str, default='Anaconda3/2022.05', metavar='name',
                    help="""Name of the Anaconda software module to load when --conda is specified""")
parser.add_argument('-p', '--partition', type=str, default='short', metavar='name',
                    help="Slurm partition name. See BMRC help for a list of available partitions")
parser.add_argument('-c', '--cpus-per-task', type=int, default=1, metavar='N',
                    help="Number of CPU cores to request for each task")
parser.add_argument('-m', '--mem-per-cpu', type=int, default=None, metavar='N',
                    help="""Amount of memory to allocate per CPU core (in megabytes). 
                         Will use BMRC default if not specified.""")
parser.add_argument('-b', '--batch', type=int, default=1, metavar='N',
                    help="""Number of commands to execute per task. 
                    For example: if cmdTxt contains 10 commands, setting this to 2 will result in 5 tasks with 
                    2 commands executed per task. Commands within a batch are executed in series""")
parser.add_argument('-L', '--log-env', action='store_true', default=False,
                    help="Log pip environment info (pip list, printenv) to logdir")
parser.add_argument('-v', nargs='+', default=None, metavar='var',
                    help="Specify environment variables to export to the task (e.g. -v HOME MY_VAR=1)")

args = parser.parse_args()

# Checks and argument prep
# Check log directory exists
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir, exist_ok=True)

# Make environment robust to whether or not trailing dash is used (ivybridge or -ivybridge)
if args.venv:
    if args.venv[-1] != "-":
        args.venv = args.venv + "-"
    # Check environment exists for sklake at least
    if not os.path.exists(args.venv + "skylake"):
        print('Check project virtual environment path is written correctly:', args.venv)
        sys.exit(-1)


# Count how many files there are to process
w = open(args.cmdsTxt, 'r')
numCmds = 0
for line in w:
    numCmds += 1
w.close()

# write submission script
## Make runName and scriptName
runName = args.cmdsTxt.split('.')[0].replace('/', '.')
logfile = os.path.join(args.logdir, runName + '-%j.log')  # runName-(job id).log
scriptName = args.cmdsTxt.split('.')[0] + '.sh'

## Open script for writing
w = open(scriptName, 'w')

## Basic arguments
w.write('#!/bin/bash\n')
w.write(f'#SBATCH -J {runName}\n')  # name of job (based on name of cmdsTxt)
w.write('#SBATCH -A doherty.prj\n')  # project name
w.write(f'#SBATCH -p {args.partition}\n')  # slurm partition
if numCmds > 1:
    logfile = os.path.join(args.logdir, runName + '-%A_%a.log')  # runName-(job id)_(array index).log
    w.write(f'#SBATCH --array 1-{numCmds}:{args.batch}\n')  # line to indicate array job
w.write(f'#SBATCH -o {logfile}\n')  # output file for std and err output stream
w.write(f'#SBATCH --cpus-per-task {args.cpus_per_task}\n')  # number of cpus
if args.mem_per_cpu:
    w.write(f'#SBATCH --mem-per-cpu {args.mem_per_cpu}\n')  # amount of memory
w.write('#SBATCH -D ./\n')  # sets working directory to submitter's directory
w.write('#SBATCH --requeue\n')
w.write('#SBATCH --time 600\n')

# export custom environment variables
if args.v:
    export = ','.join(args.v)
    w.write(f'#SBATCH --export {export}\n')

## Write main commands
w.write('\n')
w.write('module load Python/3.7.4-GCCcore-8.3.0\n') # load version of python
w.write('CPU_ARCHITECTURE=$(/apps/misc/utils/bin/get-cpu-software-architecture.py)\n') # determine Ivybridge or Skylake compatibility on this node
w.write('if [[ ! $? == 0 ]]; then\n\techo "Fatal error: Please send the following information to the BMRC team: Could not determine CPU software architecture on $(hostname)"\n\texit 1\nfi\n') # Error handling if CPU architecture not determined

## Source virtual environment if provided
if args.venv:
    w.write('source ' + args.venv + '${CPU_ARCHITECTURE}/bin/activate\n') # Script is now run within this pip virtual environment
elif args.conda:
    w.write(f'module load {args.conda_module}\n')
    w.write('eval "$(conda shell.bash hook)"\n')
    w.write(f'conda activate {args.conda}\n')

## Write further commands
w.write('export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}\n')  # disallow numpy multithreading
w.write('\n')
w.write('SECONDS=0\n') # to measure CPU time
w.write('echo $(date +%d/%m/%Y\ %H:%M:%S)\n') # echo start date
w.write('\n')

## specify input/output
w.write('cmdList="' + args.cmdsTxt + '"\n')

## read line number SLURM_ARRAY_TASK_ID and execute it; one-liner: sed -n ${SLURM_ARRAY_TASK_ID}p $cmdList | bash
if numCmds > 1:
    cmd = (
        'cmd=$(sed -n ${SLURM_ARRAY_TASK_ID}p $cmdList)\n'
        'echo $cmd\n'
        'bash -c "$cmd"\n'
    )

    if args.batch > 1:
        batch_cmd = (
            '\n'
            f'last_index={numCmds}\n'
            'last_array_task_id=$(( SLURM_ARRAY_TASK_ID + SLURM_ARRAY_TASK_STEP - 1 ))\n'
            'if [ "${last_index}" -lt "${last_array_task_id}" ]\n'
            'then\n'
            'last_array_task_id="${last_index}"\n'
            'fi\n\n'

            'while [ "${SLURM_ARRAY_TASK_ID}" -le "${last_array_task_id}" ]\n'
            'do\n'
            'echo $(date): starting work on SLURM_ARRAY_TASK_ID=$(printenv SLURM_ARRAY_TASK_ID)\n\n'

            f'{cmd}\n'

            'export SLURM_ARRAY_TASK_ID=$(( SLURM_ARRAY_TASK_ID + 1 ))\n'
            'done\n'
        )
        w.write(batch_cmd)
    else:
        w.write(cmd)
else:
    wCmd = open(args.cmdsTxt, 'rU')
    cmdLine = wCmd.read()
    wCmd.close()
    w.write(cmdLine + '\n')
    
w.write('\n')
w.write('duration=$SECONDS\n')
w.write('echo "CPU time $pheno: $(($duration / 60)) min $((duration % 60)) sec"\n')
w.write('echo $(date +%d/%m/%Y\ %H:%M:%S)\n')
w.write('echo \'\'')

if args.log_env:
    # write files describing packages used (pip environment) and full run environment
    w.write('pip list > "' + args.logdir + runName + '_${SLURM_JOB_ID}_"`date +%Y-%m-%d-%H%M%S`"_pip_env.txt"\n')
    w.write('printenv > "' + args.logdir + runName + '_${SLURM_JOB_ID}_"`date +%Y-%m-%d-%H%M%S`"_run_env.txt"\n')

w.close()
print('\nBMRC sbatch script written to:', scriptName)