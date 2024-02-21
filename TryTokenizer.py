import tiktoken

"""
'gpt2' = {function} <function gpt2 at 0x000002CA9FC709A0>
'r50k_base' = {function} <function r50k_base at 0x000002CA9FC70E00>
'p50k_base' = {function} <function p50k_base at 0x000002CA9FC70CC0>
'p50k_edit' = {function} <function p50k_edit at 0x000002CA9FC70D60>
'cl100k_base' = {function} <function cl100k_base at 0x000002CAA0F48F40>
"""

tokenizer = tiktoken.get_encoding("p50k_edit")
# assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4")


# {str} '<|fim_prefix|>'
# {str} '<|endofprompt|>'
# {str} '<|endoftext|>'
# {str} '<|fim_suffix|>'
# {str} '<|fim_middle|>'

encoded = tokenizer.encode("<|endoftext|><|fim_prefix|>", allowed_special="all")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
