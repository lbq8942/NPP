The source code of paper "Modeling Neural Point Process by Attention Mechanism and Intensity-free Distribution".


### 1.Dataset
Six datasets are available at [here](https://drive.google.com/drive/folders/11gQboMe7nRR7Xb-hO-kZA3eWkqVh_Ezq). Download and put them in folder *data*. 

### 2.Running
Before running the following command, modify the parameter "pro_path" in *args.py* as your *NPP* path.

```powershell
python main.py   --data hawkes_ind  --gpu 1 --flownum 2 --bs 64 --patience 10 --hdim 256 
python main.py   --data hawkes_dep1  --gpu 1 --flownum 4  --bs 256 --patience 20 --hdim 256 
python main.py   --data hawkes_dep_m3  --gpu 1 --flownum 5 --flowlen 1  --bs 64 --patience 10 --hdim 256 
python main.py --data hawkes_dep_m5   --gpu 1  --map 0 --flownum 5 --flowlen 1  --bs 10 --patience 10 --hdim 256
python main.py  --data stack_overflow   --gpu 1 --basis 1 –map 1 --flownum 4 --flowlen 1  --bs 64 --patience 10 --hdim 256 
python main.py  --data mooc --gpu 1 --map 2 --flownum 4 --flowlen 1  --bs 64 --patience 10 --hdim 256 
```


