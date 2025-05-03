import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

class InferenceRunner:
    def __init__(self, model_name="defog/sqlcoder-34b-alpha"):
        self.tokenizer, self.model = self.get_tokenizer_model(model_name)

    @staticmethod
    def generate_prompt(question, prompt_file="prompt.md", metadata_file="metadata.sql", instruction_file="instruction.md", knowledge="knowledge"):
        with open(prompt_file, "r") as f:
            prompt = f.read()
        with open(metadata_file, "r") as f:
            table_metadata_string = f.read()
        with open(instruction_file, "r") as f:
            instruction = f.read()
        prompt = prompt.format(
            user_question=question, table_metadata_string=table_metadata_string, instruction=instruction, knowledge=knowledge
        )
        return prompt

    @staticmethod
    def get_tokenizer_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            use_cache=True,
        )
        return tokenizer, model

    def run_inference(self, question, prompt_file="prompt.md", metadata_file="metadata.sql", instruction_file="instruction.md", knowledge="knowledge"):
        prompt = self.generate_prompt(question, prompt_file, metadata_file, instruction_file, knowledge)
        # print(prompt)
        eos_token_id = self.tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=False,
            return_full_text=False,
            num_beams=5,
        )
        generated_query = (
            pipe(
                prompt,
                num_return_sequences=1,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id,
            )[0]["generated_text"]
            .split(";")[0]
            .split("```")[0]
            .strip()
            + ";"
        )
        return generated_query

if __name__ == "__main__":
    _default_question = "Do we get more sales from customers in New York compared to customers in San Francisco? Give me the total sales for each city, and the difference between the two."
    parser = argparse.ArgumentParser(description="Run inference on a question")
    parser.add_argument("-q", "--question", type=str, default=_default_question, help="Question to run inference on")
    args = parser.parse_args()
    question = args.question
    print("Loading a model and generating a SQL query for answering your question...")
    runner = InferenceRunner()
    print(runner.run_inference(question))