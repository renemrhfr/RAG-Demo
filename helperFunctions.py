from transformers import AutoTokenizer, TFAutoModel, AutoModel
import tensorflow as tf

model_name = "danielheinz/e5-base-sts-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)


def generate_embedding(text):
    """
    Generates an Embedding from the provided Text using the
    Tokenizer that is provided in this Script.
    """
    encoded_input = tokenizer(
        text, padding=True, truncation=True, return_tensors='tf')
    with tf.GradientTape() as tape:
        outputs = model(encoded_input)
        embeddings = outputs.last_hidden_state
    return tf.reduce_mean(embeddings, axis=1)


def count_tokens(paragraph, max_token_length):
    """
    Checks if "paragraph" is exceeding the max_token_length. 
    """
    tokens = tokenizer.tokenize(paragraph)
    token_count = len(tokens)
    if token_count > max_token_length:
        return False
    return True

def cosine_similarity(a, b):
    """
    Calculates the Cosine Similarity of two provided Embeddings.
    """
    normalize_a = tf.nn.l2_normalize(a, axis=-1)
    normalize_b = tf.nn.l2_normalize(b, axis=-1)
    cos_similarity = tf.reduce_sum(
        tf.multiply(normalize_a, normalize_b), axis=-1)
    return cos_similarity.numpy()