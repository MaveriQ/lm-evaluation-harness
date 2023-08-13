from typing import Optional, Union
import transformers
import torch
from lm_eval.models.gpt2 import HFLM, _get_dtype
from morphpiece import MorphPieceBPE

class MorphGPT(HFLM):
    def __init__(
        self,
        device="cuda",
        pretrained="maveriq/morphgpt-base",
        low_cpu_mem_usage=None,
        subfolder=None,
        revision="main",
        tokenizer=None,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
        use_auth_token: Optional[bool] = True,
        dtype: Optional[Union[str, torch.dtype]]="auto",
        batch_size=1,
    ):
        # super().__init__(
        #     device=device,
        #     pretrained=pretrained,
        #     low_cpu_mem_usage=low_cpu_mem_usage,
        #     subfolder=subfolder,
        #     tokenizer=tokenizer,
        #     max_batch_size=max_batch_size,
        #     max_length=max_length,
        #     load_in_8bit=load_in_8bit,
        #     trust_remote_code=trust_remote_code,
        #     dtype=dtype,
        #     batch_size=batch_size,
        # )

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                        tokenizer,
                        transformers.PreTrainedTokenizer
                        ) or isinstance(
                        tokenizer,
                        transformers.PreTrainedTokenizerFast
                        )
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                self.tokenizer = MorphPieceBPE()

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained,
                    load_in_8bit=load_in_8bit,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    revision=revision,
                    torch_dtype=_get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    use_auth_token=use_auth_token,
                    ).to(self.device)
            self.tokenizer = MorphPieceBPE()

        else:
            raise TypeError('Parameter pretrained should be of type str or transformers.PreTrainedModel')

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

if __name__=="__main__":
    model = MorphGPT()
    print(model.tokenizer.encode("hello"))
    
    tokenizer = MorphPieceBPE()
    print(tokenizer.tokenize("hello"))