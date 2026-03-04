import faulthandler, sys, os, signal
faulthandler.enable()
# Also dump on SIGSEGV
faulthandler.register(signal.SIGUSR1)

sys.path.insert(0, '.')
os.chdir('/home/samir/Desktop/WORK/f1tenth_gym_ws')
sys.argv = ['run_mcts_known_dyn_experiments.py', '--max-steps', '10000', '--render']

import runpy
runpy.run_path('run_mcts_known_dyn_experiments.py', run_name='__main__')