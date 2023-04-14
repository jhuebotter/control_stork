from control_stork.optimizers import SMORMS3
from control_stork.nodes import CellGroup
from control_stork.nodes import StaticInputGroup, ReadoutGroup
from control_stork.connections import Connection
from control_stork.models import RecurrentSpikingModel

if __name__ == '__main__':

    n_hidden = 3
    input_dim = 10
    hidden_dim = 265
    output_dim = 10
    
    rsm = RecurrentSpikingModel()

    prev = rsm.add_group(StaticInputGroup(input_dim, name='Static Input Group'))
    for i in range(n_hidden):
        new = rsm.add_group(CellGroup(hidden_dim, name=f'Hidden Cell Group {i+1}'))    
        rsm.add_connection(Connection(prev, new, bias=True))
        prev = new
    new = rsm.add_group(ReadoutGroup(output_dim, name='Readout Group'))  
    rsm.add_connection(Connection(prev, new, bias=True))

    rsm.summary()