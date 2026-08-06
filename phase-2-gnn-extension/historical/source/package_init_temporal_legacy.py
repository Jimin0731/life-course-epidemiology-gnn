# Import original models (without temporal encoding)
#from .models_gnn import EconomicGNN as EconomicGNN_Original
#from .hybridgnn import HybridGNN as HybridGNN_Original
#from .models_weightedgat import WeightedGATGNN as WeightedGATGNN_Original

# Import temporal-aware models (with LSTM/GRU/CNN temporal encoding)
from .models_gnn_temporal import EconomicGNN
from .hybridgnn_temporal import HybridGNN
from .models_weightedgat_temporal import WeightedGATGNN

# Import temporal encoders separately if needed
from .temporal_encoders import LSTMEncoder, GRUEncoder, Conv1DEncoder, TransformerEncoder
