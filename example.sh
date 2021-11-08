# ENZYMES
python -m train --datadir=data --bmname=ENZYMES --cuda=3 --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --cuda=1 --num-classes=6 --method=soft-assign

# DD
python -m train --datadir=data --bmname=DD --cuda=0 --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
python -m train --bmname=DD --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --cuda=1 --num-classes=2 --method=soft-assign

# ACP
python -m train --datadir=data --bmname=ACP --train-ratio=0.75 --max-nodes=200 --epochs=100 --num-gc-layers=3 --dropout=0.8 --num-classes=2
# ACP
python -m train --datadir=data --bmname=ACP --max-nodes=200 --epochs=100 --num-gc-layers=3 --num-classes=2
# ACP - Diffpool
python -m train --bmname=ACP --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --num-classes=2 --method=soft-assign
