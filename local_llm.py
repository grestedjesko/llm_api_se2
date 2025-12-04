import asyncio
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        print(f"Загрузка модели {model_name}...")
 
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Используется Apple MPS")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Используется CUDA")
        else:
            self.device = "cpu"
            print("Используется CPU")
 
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
 
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": self.device},
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()

    def _build_prompt_text(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_chat_sync(
        self,
        messages,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Синхронная версия генерации для внутреннего использования."""
        prompt_text = self._build_prompt_text(messages)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start_time

        generated = outputs[0][input_len:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        completion_tokens = generated.shape[0]
        prompt_tokens = int(input_len)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "text": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
        }

    async def generate_chat(
        self,
        messages,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Асинхронная генерация чата."""
        return await asyncio.to_thread(
            self._generate_chat_sync,
            messages,
            max_new_tokens,
            temperature,
            top_p,
        )

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str = "Ты полезный русскоязычный ассистент.",
    ) -> str:
        """Асинхронная генерация текста."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        result = await self.generate_chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return result["text"]


async def main():
    llm = LocalLLM("Qwen/Qwen3-4B")

    tests = [
        "Объясни простыми словами, что такое квантовые компьютеры.",
        "Напиши короткое стихотворение о ветре.",
        "Как приготовить хороший омлет?"
    ]

    for t in tests:
        print("\nПромпт:", t)
        result = await llm.generate(t)
        print("Ответ:", result)


if __name__ == "__main__":
    asyncio.run(main())
