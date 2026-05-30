"""
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * https://github.com/martiansideofthemoon/ai-detection-paraphrases.git
 """

import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()

    def paraphrase_batch(self, input_texts, lex_diversity, order_diversity, prefixes=None, sent_interval=3, **kwargs):
        """Paraphrase a batch of texts using the DIPPER model.

        Args:
            input_texts (List[str]): List of texts to paraphrase.
            lex_diversity (int): Lexical diversity (0–100).
            order_diversity (int): Order diversity (0–100).
            prefixes (List[str] or None): List of prefixes (same length as input_texts) or a single prefix applied to all.
            sent_interval (int): How many sentences to include per generation block.
            **kwargs: Additional generation parameters like top_p, top_k, max_length.

        Returns:
            List[str]: List of paraphrased outputs.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100]
        assert order_diversity in [0, 20, 40, 60, 80, 100]

        if isinstance(prefixes, str):
            prefixes = [prefixes] * len(input_texts)
        elif prefixes is None:
            prefixes = [""] * len(input_texts)

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity

        final_inputs = []
        new_prefixes = []

        for input_text, prefix in zip(input_texts, prefixes):
            input_text = " ".join(input_text.split())
            sentences = sent_tokenize(input_text)
            prefix = " ".join(prefix.replace("\n", " ").split())
            output_text = ""
            local_prefix = prefix

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
                final_input_text = f"lexical = {lex_code}, order = {order_code}"
                if local_prefix:
                    final_input_text += f" {local_prefix}"
                final_input_text += f" <sent> {curr_sent_window} </sent>"
                final_inputs.append(final_input_text)
                new_prefixes.append(local_prefix)

        tokenized_inputs = self.tokenizer(final_inputs, return_tensors="pt", padding=True, truncation=True)

        if torch.cuda.is_available():
            tokenized_inputs = {k: v.cuda() for k, v in tokenized_inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**tokenized_inputs, **kwargs)

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Combine output_texts according to original batch sizes
        output_texts = [""] * len(input_texts)
        prefix_idx = [0] * len(input_texts)
        chunk_counts = [len(sent_tokenize(text)) // sent_interval + 1 for text in input_texts]

        idx = 0
        for i, count in enumerate(chunk_counts):
            output_chunks = decoded_outputs[idx:idx + count]
            output_texts[i] = " ".join(output_chunks)
            idx += count

        return output_texts


if __name__ == "__main__":
    dp = DipperParaphraser(model="kalpeshk2011/dipper-paraphraser-xxl")

    prompt = "Tracy is a fox."
    input_text = "It is quick and brown. It jumps over the lazy dog."

    output_l60_o60_greedy = dp.paraphrase(input_text, lex_diversity=80, order_diversity=60, prefix=prompt, do_sample=False, max_length=512)
    output_l60_sample = dp.paraphrase(input_text, lex_diversity=80, order_diversity=0, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=512)
    print(f"Input = {prompt} <sent> {input_text} </sent>\n")
    print(f"Output (Lexical diversity = 80, Order diversity = 60, Greedy) = {output_l60_o60_greedy}\n")
    print(f"Output (Lexical diversity = 80, Sample p = 0.75) = {output_l60_sample}\n")
    print("--------------------\n")

    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."
    input_text = "They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns."

    print(f"Input = {prompt} <sent> {input_text} </sent>\n")
    output_l60_sample = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, prefix=prompt, do_sample=False, top_p=0.75, top_k=None, max_length=512)
    print(f"Output (Lexical diversity = 60, Sample p = 0.75) = {output_l60_sample}\n")