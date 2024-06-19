import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
from transformers import BartConfig, BartTokenizer, BertTokenizer
import sys, os, warnings
warnings.filterwarnings("ignore")

from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "atomic-thunder-15-7.dat")

params = load_params(MODEL_PATH)
params = jax.tree_map(np.asarray, params)

tokenizer_en = BartTokenizer.from_pretrained('facebook/bart-base')
tokenizer_yue = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

config = BartConfig.from_pretrained('Ayaka/bart-base-cantonese')
generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)

# generate

sentences = [
    'anaemia',
    'Get out!',
]
# inputs = tokenizer_en(sentences, return_tensors='jax', padding=True)
# src = inputs.input_ids.astype(np.uint16)
# mask_enc_1d = inputs.attention_mask.astype(np.bool_)
# mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

# encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
# generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=128)

# decoded_sentences = tokenizer_yue.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# for sentence in decoded_sentences:
#     sentence = sentence.replace(' ', '')
#     print(sentence)

def transcan_translate(content:list[str]) -> list[str]:
    inputs = tokenizer_en(content, return_tensors="jax", padding=True)
    src = inputs.input_ids.astype(np.uint16)
    mask_enc_1d = inputs.attention_mask.astype(np.bool_)
    mask_enc = np.einsum("bi,bj->bij", mask_enc_1d, mask_enc_1d)[:, None]

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generate_ids = generator.generate(
        encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=128
    )

    decoded_sentences = tokenizer_yue.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # res = ""
    # for sentence in decoded_sentences:
    #     sentence = sentence.replace(" ", "")
    #     logger.debug(sentence)
    #     res += sentence

    return decoded_sentences

decoded_sentences = transcan_translate(sentences)
for sentence in decoded_sentences:
    sentence = sentence.replace(' ', '')
    print(sentence)
