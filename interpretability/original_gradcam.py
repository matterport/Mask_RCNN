"""
Core Module for Grad CAM Algorithm
Adapted from tf-explain
"""
import cv2
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from tf_explain.core import GradCAM
from tensorflow.keras.layers import Wrapper

from tf_explain.utils.display import grid_display, image_to_uint_255


def KRe_map():
    """
    Returns a MatPlotLib colormap that goes from black (K) to red (Re).

    The colors are defined as:
    0% black, 20% black, 100% red

    Visual example:
    https://colorzilla.com/gradient-editor/#000000+0,000000+20,ff0000+100
    :return: MatPlotLib colormap
    """
    black_stop = 0.2
    red_stop = 1.0

    color_dict = {
        "red": (
        (0.0, 0.0, 0.0), (black_stop, 0.0, 0.0), (red_stop, 1.0, 1.0),),
        "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),),
        "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),),
    }
    return LinearSegmentedColormap("KRe", color_dict)


class OriginalGradCAM(GradCAM):
    """
    Perform Grad CAM algorithm for a given input

    This version of GradCAM is more true to the original paper (below) as it uses ReLU
    on the CAMs instead of on the gradients during the backpropagation.

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(
            self,
            validation_data,
            model,
            class_index,
            layer_name=None,
            colormap=None,
            image_weight=0.7,
            use_guided=False,
            original_images=None,
            distance_from_end=1,
    ):
        """
        Compute GradCAM for a specific class index.

        Changes from tf_explain.core.grad_cam.explain:
            ADDED use_guided parameter
            ADDED original_images parameter
            ADDED distance_from_end
            CHANGED colormap default

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            layer_name (str): Targeted layer for GradCAM. If no layer is provided, it is
                automatically inferred from the model architecture. Based on
                distance_from_end if specified.
            class_index (int): Index of targeted class [c in GradCAM-Selvaraju paper]
            colormap (int): OpenCV Colormap to use for heatmap visualization
            use_guided (boolean): whether to use guided ReLU for gradient
                backpropagation
            original_images (list): original images from the validation data
                without preprocessing applied for use with heatmap overlay
            distance_from_end (int): Uses the ith layer from the back that is suitable
               for GradCAM as targeted layer. Overridden by layer_name.
            image_weight (float): An optional `float` value in range [0,1] indicating
               the weight of the input image to be overlaying the calculated attribution
                maps. Defaults to `0.7`.

        Returns:
            numpy.ndarray: Grid of all the GradCAM
        """
        images, _ = validation_data
        if original_images is None:
            original_images = images
        else:
            assert len(images) == len(original_images)

        if layer_name is None:
            layer_name = self.infer_grad_cam_target_layer(model,
                                                          distance_from_end)

        """
        Reference to GradCAM-Selvaraju paper [requires use_guided == False]
            outputs = A [feature map activations for all feature maps]
            grads = frac{partial y^c}{partial A}
            cams = sum_k( alpha^c_k A^k )
            relu_cams = ReLU(cams) = L^c_{Grad-CAM}
        """
        outputs, grads = self.get_gradients_and_filters(
            model, images, layer_name, class_index, use_guided
        )

        """
        Compute weighted combination of forward activation maps
         [sum_k( alpha^c_k A^k )]

        'Ponderated' is fancy (French-style English?) vocabulary for 'weighted'.

        Reference to GradCAM-Selvaraju paper
            # Global Average Pooling of gradients grads
            #   weights = alpha^c_k
            weights = tf.reduce_mean(grad, axis=(0, 1))
            # Multiply weights and outputs and sum over k (axis -1, the filters)
            #   cam = sum_k( alpha^c_k A^k ) = sum_k( weights A^k ) 
            cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        """
        cams = self.generate_ponderated_output(outputs, grads)

        # Apply ReLU to the weighted combination of forward activation maps
        # [ReLU(cams) = L^c_{Grad-CAM}]
        relu_cams = [np.clip(cam.numpy(), a_min=0, a_max=None) for cam in cams]

        heatmaps = np.array(
            [
                # heatmap_display resizes CAM to the image size, then scales values to
                #  the range 0 to 1 where 0 is assigned to the minimal value in the CAM
                #  and 1 to the maximum value. Then all values are multiplied with 255
                #  to convert to a grayscale channel which is used as input for CV2
                #  linear colormap.
                self.heatmap_display(relu_cam, image, colormap, image_weight)
                for relu_cam, image in zip(relu_cams, original_images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    @staticmethod
    @tf.function
    def get_gradients_and_filters(
            model, images, layer_name, class_index, use_guided=False
    ):
        """
        Generate guided gradients and convolutional outputs with an inference.

        Changes from tf_explain.core.grad_cam.get_gradients_and_filters:
            ADDED use_guided parameter

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            layer_name (str): Targeted layer for GradCAM
            class_index (int): Index of targeted class
            use_guided (boolean): whether to use guided ReLU for gradient
                backpropagation

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
        """

        # grad_model outputs the feature maps [A] at the target layer and
        #  the prediction [y]
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        """
        Reference to GradCAM-Selvaraju paper
            conv_outputs = A [feature map activations for all feature maps]
            grads = frac{partial y^c}{partial A}
            predictions = y
            loss = y^c
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss,
                              conv_outputs)  # target=loss, source=conv_outputs

        if use_guided:
            # Use Guided ReLU, see https://arxiv.org/abs/1412.6806
            guided_grads = (
                    tf.cast(conv_outputs > 0, "float32")
                    * tf.cast(grads > 0, "float32")
                    * grads
            )
            return conv_outputs, guided_grads

        return conv_outputs, grads

    @staticmethod
    def infer_grad_cam_target_layer(model, distance_from_end=1):
        """
        Search for the last convolutional layer to perform Grad CAM, as stated
        in the original paper.

        Changes from tf_explain.core.grad_cam.infer_grad_cam_target_layer:
            ADDED distance_from_end parameter

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            distance_from_end (int): Uses the ith layer from the back that is suitable
               for GradCAM as targeted layer.

        Returns:
            str: Name of the target layer
        """
        distance = 0
        for layer in reversed(model.layers):
            if isinstance(layer, Wrapper):
                layer = layer.layer
            # Select closest 4D layer to the end of the network.
            if len(layer.output_shape) == 4:
                if distance == distance_from_end:
                    return layer.name
                else:
                    distance += 1

        raise ValueError(
            "Model does not seem to contain 4D layer. Grad CAM cannot be applied."
        )

    @staticmethod
    def heatmap_display(heatmap_small, original_image, colormap=None,
                        image_weight=0.7):
        """
        Apply a heatmap (as an np.ndarray) on top of an original image.

        Changes from tf_explain.utils.display.heatmap_display:
            CHANGED colormap default
            CHANGED renamed heatmap parameter to heatmap_small
            CHANGED scaling of heatmap w.r.t. colormap

        Args:
            heatmap_small (numpy.ndarray): Array corresponding to the heatmap
            original_image (numpy.ndarray): Image on which we apply the heatmap
            colormap (int): OpenCV Colormap to use for heatmap visualization
            image_weight (float): An optional `float` value in range [0,1] indicating
                the weight of the input image to be overlaying the calculated
                attribution maps. Defaults to `0.7`.

        Returns:
            np.ndarray: Original image with heatmap applied
        """
        heatmap_large = cv2.resize(
            heatmap_small, (original_image.shape[1], original_image.shape[0])
        )

        image = image_to_uint_255(original_image)

        # Scale heatmap relative to the absolute maximum
        heatmap_large = heatmap_large / np.abs(heatmap_large).max()

        if colormap is None:
            # By default we use the custom KRe colormap
            colormap = OriginalGradCAM.cmap_mpl2cv2(KRe_map())

        heatmap = cv2.applyColorMap(
            cv2.cvtColor((heatmap_large * 255).astype("uint8"),
                         cv2.COLOR_GRAY2BGR),
            colormap,
        )

        output = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_weight, heatmap, 1, 0
        )

        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    @staticmethod
    def cmap_mpl2cv2(c_map, rgb_order=False):
        """
        Extract colormap color information as a LUT compatible with cv2.applyColormap().
        Default channel order is BGR.

        SOURCE https://gitlab.com/cvejarano-oss/cmapy/-/blob/master/cmapy.py
        MIT LICENSE

        Args:
            c_map: MatPlotLib colormap
            rgb_order: boolean, if false or not set, the returned array will be in
                       BGR order (standard OpenCV format). If true, the order
                       will be RGB.

        Returns:
            A numpy array of type uint8 containing the colormap.
        """
        rgba_data = matplotlib.cm.ScalarMappable(cmap=c_map).to_rgba(
            np.arange(0, 1.0, 1.0 / 256.0), bytes=True
        )
        rgba_data = rgba_data[:, 0:-1].reshape((256, 1, 3))

        # Convert to BGR (or RGB), uint8, for OpenCV.
        cmap = np.zeros((256, 1, 3), np.uint8)

        if not rgb_order:
            cmap[:, :, :] = rgba_data[:, :, ::-1]
        else:
            cmap[:, :, :] = rgba_data[:, :, :]

        return cmap
