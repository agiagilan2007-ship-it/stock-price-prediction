import tensorflow as tf

def build_lstm_model(input_shape, horizon=7):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(horizon, name='pred')(x)  # predict multi-step closes
    model = tf.keras.Model(inputs, outputs, name='lstm_model')
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Simple Transformer encoder block for time series
def build_transformer_model(input_shape, horizon=7, d_model=64, num_heads=4, ff_dim=128, num_layers=2):
    inputs = tf.keras.Input(shape=input_shape)  # (window, features)
    x = tf.keras.layers.Dense(d_model)(inputs)
    # positional encoding (learned)
    pos_emb = tf.keras.layers.Embedding(input_dim=input_shape[0], output_dim=d_model)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    x += pos_emb(positions)

    for _ in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff(x))

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(horizon, name='pred')(x)
    model = tf.keras.Model(inputs, outputs, name='transformer_model')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
