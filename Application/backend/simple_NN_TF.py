import tensorflow as tf
from tensorflow.keras import layers # type: ignore
import tf2onnx
import onnx

# Not possible to use classes due to subclassing, function is used instead
def create_model():
    inputs = tf.keras.Input(shape=(2,))
    x = layers.Dense(4, activation='relu')(inputs)
    x = layers.Dense(6, activation='sigmoid')(x)
    x = layers.Dense(14, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(4)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model






if __name__ == "__main__":
    model = create_model()
    model.compile(optimizer='adam', loss='mse')
    input_data = tf.random.normal((1, 2))
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec((1, 2), tf.float32)])
    onnx.save(onnx_model, "./NNVct/Application/backend/model_TF.onnx")