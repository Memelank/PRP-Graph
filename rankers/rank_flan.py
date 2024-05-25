import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
class FlanT5():
    def __init__(
        self,
        model:str,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.model = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16, device_map="auto")
        self.prompt = '''Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A:{doc1}

Passage B:{doc2}

Output Passage A or Passage B:'''

    def generate(self, query, passage_a, passage_b, max_length=4096):
        query = self.truncate(query, 32)
        passage_a = self.truncate(passage_a, 128)
        passage_b = self.truncate(passage_b, 128)
        input_text = self.prompt.format(query=query, doc1=passage_a, doc2=passage_b)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {i}'
                                                                for i in ["A", "B"]],
                                                                return_tensors="pt",
                                                                add_special_tokens=False,
                                                                padding=True).input_ids[:, -1]
        decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                return_tensors="pt",
                                                add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits[0][-1]
            distributions = torch.softmax(logits, dim=0)
            scores = distributions[target_token_ids]
            ranked_1 = sorted(zip(["A", "B"], scores), key=lambda x: x[1], reverse=True)
        
        input_text = self.prompt.format(query=query, doc1=passage_b, doc2=passage_a)
        
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {i}'
                                                                for i in ["A", "B"]],
                                                                return_tensors="pt",
                                                                add_special_tokens=False,
                                                                padding=True).input_ids[:, -1]
        decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                return_tensors="pt",
                                                add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits[0][-1]
            distributions = torch.softmax(logits, dim=0)
            scores = distributions[target_token_ids]
            ranked_2 = sorted(zip(["A", "B"], scores), key=lambda x: x[1], reverse=True)

        score_1 = ranked_1[0][1] if ranked_1[0][0] == "A" else ranked_1[1][1]
        score_2 = ranked_2[0][1] if ranked_2[0][0] == "A" else ranked_2[1][1]
        output = [score_1.item(), score_2.item()]
        
        return output

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])