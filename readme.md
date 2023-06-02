
Dataset has not been upload yet.

python main.py   --data hawkes_ind  --gpu 1 --flownum 2 --bs 64 --patience 10 --hdim 256 >/dev/null 2>&1 &

python main.py   --data hawkes_dep1  --gpu 1 --flownum 4  --bs 256 --patience 20 --hdim 256 >/dev/null 2>&1 &

python main.py   --data hawkes_dep_m3  --gpu 1 --flownum 5 --flowlen 1  --bs 64 --patience 10 --hdim 256 >/dev/null 2>&1 &

python main.py --data hawkes_dep_m5   --gpu 1  --map 0 --flownum 5 --flowlen 1  --bs 10 --patience 10 --hdim 256 >/dev/null 2>&1 &

python main.py  --data stack_overflow   --gpu 1 --basis 1 –map 1 --flownum 4 --flowlen 1  --bs 64 --patience 10 --hdim 256 >/dev/null 2>&1 &

python main.py  --data mooc --gpu 1 --map 2 --flownum 4 --flowlen 1  --bs 64 --patience 10 --hdim 256 >/dev/null 2>&1 &


