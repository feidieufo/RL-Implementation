import tensorflow as tf
from tensorflow.python.keras import layers


def nature_dqn(num_actions):
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, [8, 8], strides=4, input_shape=(84, 84, 4), activation="relu"))
    model.add(layers.Conv2D(64, [4, 4], strides=2, activation="relu"))
    model.add(layers.Conv2D(64, [3, 3], strides=1, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(num_actions))
    
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss="mse",
                  metrics=["mae"])
    
    model_target = tf.keras.Sequential()
    model_target.add(layers.Conv2D(32, [8, 8], strides=4, input_shape=(84, 84, 4), activation="relu"))
    model_target.add(layers.Conv2D(64, [4, 4], strides=2, activation="relu"))
    model_target.add(layers.Conv2D(64, [3, 3], strides=1, activation="relu"))
    model_target.add(layers.Flatten())
    model_target.add(layers.Dense(512, activation="relu"))
    model_target.add(layers.Dense(num_actions))

    return model, model_target


def mlp_dqn(num_actions, input_shape, hidden_sizes=(400,300), activation="relu"):

    model = tf.keras.Sequential()
    model.add(layers.Dense(hidden_sizes[0], input_shape=input_shape, activation=activation))
    for i in hidden_sizes[1:-1]:
        model.add(layers.Dense(hidden_sizes[i], activation=activation))

    model.add(layers.Dense(num_actions))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                  loss="mse",
                  metrics=["mae"])

    model_target = tf.keras.Sequential()
    model_target.add(layers.Dense(hidden_sizes[0], input_shape=input_shape, activation=activation))
    for i in hidden_sizes[1:-1]:
        model_target.add(layers.Dense(hidden_sizes[i], activation=activation))

    model_target.add(layers.Dense(num_actions))

    return model, model_target

