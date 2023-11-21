import pennylane as qml
from pennylane.templates.layers import StronglyEntanglingLayers
import math


num_of_items = 128
n_wires = int(math.log2(num_of_items))
wires_list = list(range(n_wires))

dev_QRS_infinite = qml.device('default.qubit', wires=n_wires)
@qml.qnode(dev_QRS_infinite)
def QRS_infinite_circ(params):
    for wire in wires_list:
        qml.Hadamard(wire)

    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)

    return qml.probs(wires=wires_list)



dev_QRS_num_of_items = qml.device('default.qubit', wires=n_wires, shots=num_of_items)
@qml.qnode(dev_QRS_num_of_items)
def QRS_num_of_items_circ(params):
    for wire in wires_list:
        qml.Hadamard(wire)
    for p in params:
        StronglyEntanglingLayers(p, wires=wires_list)
    return qml.sample()


