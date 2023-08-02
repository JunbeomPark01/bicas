# AutoLabeling

How to use
: Your dataset must be in the `./data`  folder.  ex) `AutoLabeling/data/mydata/images/*.jpg`
: Also `classes.txt` file should be place in `./data/mydata/`. ex) `AutoLabeling/data/mydata/classes.txt`

 command 
```shell
cd autolabeling
python auto_labeling.py --dataset human -y7 ./model/Y7_human.pt -p 0.76 -a 0.4 -f 0.1
```

Ensemble (Weighted Boxes Fusion)
: To use several model [m1.pt, m2.pt, m3.pt] you can use command below
```shell
python auto_labeling.py --dataset human -y7 ./model/m1.pt -y5 ./model m1.pt m2.pt -p 0.76 -a 0.4 -f 0.1
```