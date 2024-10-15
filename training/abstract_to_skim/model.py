import tensorflow as tf
import pandas as pd
from backbone.attention_block import AttentionBlock
from backbone.embedding_layer import PositionEmbeddingLayer
from cfg import ModelCFG
from loguru import logger
from datasets.utils import text_vectorization
from datasets.transform import preprocessing_data, get_data_ready

class TransformerEncoderModel(tf.keras.Model):
    def __init__(self, embed_dim, ff_dim, num_heads, dropout_rate, vocab_size, vectorizer):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.positional_embedding = PositionEmbeddingLayer(vocab_size, embed_dim)
        self.attention_block = AttentionBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.vectorizer = vectorizer

    def call(self, input_1, input_2, input_3):
        # Text
        token_output = self.vectorizer(input_1)
        token_output = self.positional_embedding(token_output)
        token_output = self.attention_block(token_output)
        token_output = tf.keras.layers.GlobalAveragePooling1D()(token_output)
        token_output = tf.keras.layers.Dropout(self.dropout_rate)(token_output)

        # Line number
        print(input_2)
        line_number_output = tf.keras.layers.Dense(32, activation="relu", name="LineNumberLayer")(input_2)
        
        # Total line
        total_line_output = tf.keras.layers.Dense(32, activation="relu", name="TotalLineLayer")(input_3)
        combined_all = tf.keras.layers.Concatenate(name="tribrid_embed")([token_output, line_number_output, total_line_output])

        return tf.keras.layers.Dense(ModelCFG.NUM_CLASSES, activation="softmax")(combined_all)
    
if __name__ == "__main__":
    train_ds, val_ds, test_ds = preprocessing_data("20k", True)
    train_df = pd.DataFrame(train_ds)
    val_df = pd.DataFrame(val_ds)
    test_df = pd.DataFrame(test_ds)

    vectorizer = text_vectorization(train_df['text'].to_numpy())
    
    train_dataset, val_dataset, test_dataset = get_data_ready(train_ds, val_ds, test_ds)

    transformer_encoder_model = TransformerEncoderModel(
        ModelCFG.EMBED_DIM,
        ModelCFG.FF_DIM, 
        ModelCFG.NUM_HEADS,
        ModelCFG.DROPOUT_RATE,
        ModelCFG.MAX_TOKENS,
        vectorizer
    )

    result = transformer_encoder_model(tf.constant(["Hello my name is Duong"]), tf.constant([[12]]), tf.constant([[15]]))
    print(result)
