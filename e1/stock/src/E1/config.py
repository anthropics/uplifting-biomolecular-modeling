from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .tokenizer import get_tokenizer

logger = logging.get_logger(__name__)


class E1Config(PretrainedConfig):
    model_type = "E1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # type: ignore
        self,
        # Model architecture/initialization
        vocab_size=None,
        hidden_size=4096,
        intermediate_size=16384,
        gated_mlp=False,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        torch_dtype="bfloat16",
        gradient_checkpointing=False,
        no_ffn_gradient_checkpointing=False,
        # Tokenization
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        # Attention implementation & rotary positional embeddings
        global_attention_every_n_layers=0,
        max_num_sequences=512,
        max_num_positions_within_seq=8192,
        max_num_positions_global=1024 * 128,
        rope_theta_within_seq=10000.0,
        rope_theta_global=100000.0,
        clip_qkv=None,
        **kwargs,
    ) -> None:
        tokenizer = get_tokenizer()
        super().__init__(
            pad_token_id=tokenizer.token_to_id("<pad>"),
            bos_token_id=tokenizer.token_to_id("<bos>"),
            eos_token_id=tokenizer.token_to_id("<eos>"),
            tie_word_embeddings=tie_word_embeddings,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        self.hidden_size = hidden_size
        if intermediate_size is None:
            intermediate_size = 3 * hidden_size if gated_mlp else 4 * hidden_size
        self.intermediate_size = intermediate_size
        self.gated_mlp = gated_mlp
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_num_positions_within_seq = max_num_positions_within_seq
        self.max_num_positions_global = max_num_positions_global

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_within_seq = rope_theta_within_seq
        self.rope_theta_global = rope_theta_global
        self.max_num_sequences = max_num_sequences
        assert clip_qkv is None or clip_qkv > 0
        self.clip_qkv = clip_qkv
        self.global_attention_every_n_layers = global_attention_every_n_layers

        self.vocab_size = tokenizer.get_vocab_size()
        self.gradient_checkpointing = gradient_checkpointing
        self.no_ffn_gradient_checkpointing = no_ffn_gradient_checkpointing

        if vocab_size is not None:
            if vocab_size < self.vocab_size:
                logger.warning(
                    f"Using vocab_size {vocab_size} smaller than {self.vocab_size} from tokenizer. MAKE SURE THIS IS INTENTIONAL."
                )
                self.vocab_size = vocab_size
            elif vocab_size > self.vocab_size:
                logger.warning(f"Using vocab_size {vocab_size} instead of smaller {self.vocab_size} from tokenizer.")
                self.vocab_size = vocab_size
        if pad_token_id is not None and pad_token_id != self.pad_token_id:
            logger.warning(f"Ignoring pad_token_id. Using {self.pad_token_id} from tokenizer")
        if bos_token_id is not None and bos_token_id != self.bos_token_id:
            logger.warning(f"Ignoring bos_token_id. Using {self.bos_token_id} from tokenizer")
        if eos_token_id is not None and eos_token_id != self.eos_token_id:
            logger.warning(f"Ignoring eos_token_id. Using {self.eos_token_id} from tokenizer")
