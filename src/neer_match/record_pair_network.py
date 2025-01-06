"""
Record pair network module.

This module contains functionality for instantiating, training, and using a record
 pair networks.
"""

from neer_match.field_pair_network import FieldPairNetwork
from neer_match.similarity_map import SimilarityMap
import tensorflow as tf
import typing


class RecordPairNetwork(tf.keras.Model):
    """Record network class.

    The class creates networks for matching records from two datasets. The networks
    consist of field pair networks, constructed according to the passed similarity map,
    that are concatenated and passed through a series of hidden dense layers. The output
    layer has a sigmoid activation function.

    The depth and width of the record pair hidden layers is specified by the
    initial_record_width_scale and record_depth parameters. The width of the initial
    hidden layers is calculated multiplying the initial_record_width_scale by the number
    of field pairs (i.e., the number of associations in the similarity map, see
    :meth:`.no_associations`). The widths of the subsequent hidden layers are
    calculated dividing the initial width by the layer depth-index plus one.

    The depth and width of the field pair networks are specified by the
    initial_feature_width_scales and  feature_depths parameters (see
    :class:`.FieldPairNetwork` for more details).

    Attributes:
        similarity_map (SimilarityMap): The similarity map object.
        initial_feature_width_scales (list[int]): The initial width scales of the hidden
            layers for each field pair network.
        feature_depths (list[int]): The depths of the networks for each field pair
            network.
        initial_record_width_scale (int): The initial width scale of the hidden layers
            for the record pair network.
        record_depth (int): The depth of the record pair network.
    """

    def __init__(
        self,
        similarity_map: SimilarityMap,
        initial_feature_width_scales: typing.Union[int, list[int]] = 10,
        feature_depths: typing.Union[int, list[int]] = 2,
        initial_record_width_scale: int = 10,
        record_depth: int = 4,
        **kwargs,
    ) -> None:
        """Initialize a record network object.

        Args:
            similarity_map: The similarity map.
            initial_feature_width_scales: The initial width scales of the hidden layers
                    for each field pair network. If an integer is passed, the same scale is
                    used for all networks.
            feature_depths: The depths of the networks for each field pair network. If
                    an integer is passed, the same depth is used for all networks.
            initial_record_width_scale: The initial width scale of the hidden layers for
                    the record pair network.
            record_depth: The depth of the record pair network.
            **kwargs: Additional keyword arguments passed to parent class
                      (:class:`tensorflow.keras.Model`).
        """
        if not isinstance(similarity_map, SimilarityMap):
            raise ValueError(
                "Input similarity_map must be an instance of SimilarityMap."
            )
        if (
            not isinstance(initial_record_width_scale, int)
            or initial_record_width_scale < 1
        ):
            raise ValueError(
                "Input initial_record_width_scale must be a positive integer."
            )
        if not isinstance(record_depth, int) or record_depth < 1:
            raise ValueError("Input record_depth must be a positive integer.")
        # The remaining arguments are check in FieldPairNetwork.

        self.similarity_map = similarity_map
        self.initial_feature_width_scales = initial_feature_width_scales
        self.feature_depths = feature_depths
        self.initial_record_width_scale = initial_record_width_scale
        self.record_depth = record_depth

        no_assoc = similarity_map.no_associations()
        if isinstance(initial_feature_width_scales, int):
            initial_feature_width_scales = [initial_feature_width_scales] * no_assoc
        if isinstance(feature_depths, int):
            feature_depths = [feature_depths] * no_assoc

        super().__init__(**kwargs)

        self.field_networks = []
        for i, name in enumerate(similarity_map.association_names()):
            self.field_networks.append(
                FieldPairNetwork(
                    size=similarity_map.association_sizes()[i],
                    initial_width_scale=initial_feature_width_scales[i],
                    depth=feature_depths[i],
                    name=name,
                )
            )
        self.concat = tf.keras.layers.Concatenate()
        self.record_layers = []
        for i in range(record_depth):
            size = (initial_record_width_scale * no_assoc) // (i + 1)
            self.record_layers += [
                tf.keras.layers.Dense(
                    max(size, 2),
                    activation=tf.keras.activations.relu,
                    name=f"hidden_record_mixing_{i}",
                )
            ]
        self.record_layers += [
            tf.keras.layers.Dense(
                1,
                tf.keras.activations.sigmoid,
                name="record_classifier",
            )
        ]

    def get_config(self) -> dict:
        """Return the configuration of the network."""
        config = super().get_config().copy()
        config.update(
            {
                "similarity_map": self.similarity_map,
                "initial_feature_width_scales": self.initial_feature_width_scales,
                "feature_depths": self.feature_depths,
                "initial_record_width_scale": self.initial_record_width_scale,
                "record_depth": self.record_depth,
            }
        )
        return config

    def build(
        self,
        input_shapes: typing.Union[
            typing.List[tf.TensorShape], typing.Dict[str, tf.TensorShape]
        ],
    ) -> None:
        """Build the network."""
        field_output_shapes = []
        if isinstance(input_shapes, dict):
            input_shapes = list(input_shapes.values())

        for i, input_shape in enumerate(input_shapes):
            self.field_networks[i].build(input_shape)
            field_output_shapes.append((input_shape[0], 1))
        self.concat.build(field_output_shapes)
        input_shapes = self.concat.compute_output_shape(field_output_shapes)
        for layer in self.record_layers:
            layer.build(input_shapes)
            input_shapes = (input_shapes[0], layer.units)
        super().build(input_shapes)

    def call(
        self, inputs: typing.Union[typing.List[tf.Tensor], typing.Dict[str, tf.Tensor]]
    ) -> tf.Tensor:
        """Run the network on input."""
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        outputs = []
        for i, x in enumerate(inputs):
            outputs.append(self.field_networks[i](x))
        output = self.concat(outputs)
        for layer in self.record_layers:
            output = layer(output)
        return output

    def get_weights(self) -> typing.List[tf.Tensor]:
        """Get the weights of all layers in the network."""
        weights = []
        for field_network in self.field_networks:
            field_weights = field_network.get_weights()
            print(f"Field Network Weights: {[w.shape for w in field_weights]}")
            weights.extend(field_weights)
        for layer in self.record_layers:
            layer_weights = layer.get_weights()
            print(f"Record Layer Weights: {[w.shape for w in layer_weights]}")
            weights.extend(layer_weights)
        return weights

    def get_biases(self) -> typing.List[tf.Tensor]:
        """Get the biases of all layers in the network."""
        biases = []
        for field_network in self.field_networks:
            field_biases = [w for w in field_network.get_weights() if len(w.shape) == 1]
            print(f"Field Network Biases: {[b.shape for b in field_biases]}")
            biases.extend(field_biases)
        for layer in self.record_layers:
            layer_biases = [w for w in layer.get_weights() if len(w.shape) == 1]
            print(f"Record Layer Biases: {[b.shape for b in layer_biases]}")
            biases.extend(layer_biases)
        return biases

    def set_weights(self, weights: typing.List[tf.Tensor]) -> None:
        """Set the weights of all layers in the network."""
        weight_index = 0
        for field_network in self.field_networks:
            num_weights = len(field_network.get_weights())
            print(f"Setting weights for {field_network.name}: Expected {num_weights}, "
                  f"Shapes {[w.shape for w in field_network.get_weights()]}, "
                  f"Provided {[w.shape for w in weights[weight_index:weight_index + num_weights]]}")
            field_network.set_weights(weights[weight_index:weight_index + num_weights])
            weight_index += num_weights
        for layer in self.record_layers:
            num_weights = len(layer.get_weights())
            print(f"Setting weights for {layer.name}: Expected {num_weights}, "
                  f"Shapes {[w.shape for w in layer.get_weights()]}, "
                  f"Provided {[w.shape for w in weights[weight_index:weight_index + num_weights]]}")
            layer.set_weights(weights[weight_index:weight_index + num_weights])
            weight_index += num_weights

    def set_biases(self, biases: typing.List[tf.Tensor]) -> None:
        """Set the biases of all layers in the network."""
        bias_index = 0
        for field_network in self.field_networks:
            weights = field_network.get_weights()
            # Extract biases from weights
            num_biases = sum(1 for w in weights if len(w.shape) == 1)
            print(f"Setting biases for {field_network.name}: Expected {num_biases}, "
                  f"Shapes {[w.shape for w in weights if len(w.shape) == 1]}, "
                  f"Provided {[b.shape for b in biases[bias_index:bias_index + num_biases]]}")
            
            # Replace only the biases
            updated_weights = []
            for w in weights:
                if len(w.shape) == 1:  # It's a bias
                    updated_weights.append(biases[bias_index])
                    bias_index += 1
                else:
                    updated_weights.append(w)  # Keep the original weight
            
            # Set the updated weights back
            field_network.set_weights(updated_weights)
        
        for layer in self.record_layers:
            weights = layer.get_weights()
            # Extract biases from weights
            num_biases = sum(1 for w in weights if len(w.shape) == 1)
            print(f"Setting biases for {layer.name}: Expected {num_biases}, "
                  f"Shapes {[w.shape for w in weights if len(w.shape) == 1]}, "
                  f"Provided {[b.shape for b in biases[bias_index:bias_index + num_biases]]}")
            
            # Replace only the biases
            updated_weights = []
            for w in weights:
                if len(w.shape) == 1:  # It's a bias
                    updated_weights.append(biases[bias_index])
                    bias_index += 1
                else:
                    updated_weights.append(w)  # Keep the original weight
            
            # Set the updated weights back
            layer.set_weights(updated_weights)


