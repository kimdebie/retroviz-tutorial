import numpy as np

def get_activations(reg, X):

    """

    Get the activations from all layers in a multi-layer perceptron.

    Parameters
    ----------

    reg : sklearn model object
        Trained Multi-Layer Perceptron object.

    X : numpy array
        Input for which the activations must be generated.

    Returns
    -------

    activations : numpy array
        Activations from all layers of the network.
    """

    # obtain the hidden layer sizes
    hidden_layer_sizes = reg.hidden_layer_sizes

    # place layer sizes in a list
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)

    # add input and output shape to the list
    layer_units = [X.shape[1]] + hidden_layer_sizes + [reg.n_outputs_]

    # store the activations, beginning with the original input
    activations = [X]

    # for each layer, store the activations
    for i in range(reg.n_layers_ - 1):
        activations.append(np.empty((X.shape[0], layer_units[i + 1])))
    reg._forward_pass(activations)

    return activations
