echo "Compile" 
python sn_run.py compile --pef-name BraggNN

echo "Run"
python sn_run.py run --pef out/BraggNN/BraggNN.pef -maxep 10000
