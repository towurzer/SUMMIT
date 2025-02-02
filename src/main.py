import torch
import pandas as pd

from dataset import TranslationDataset



torch.mean(torch.zeros(4))
print(torch.zeros(4))
print(torch.mean(torch.zeros(4)))
print(type(pd.DataFrame()))

print(f'has cuda? {torch.cuda.is_available()}')


#get input
val = input("Enter your value: ")
print(val)
#load model


#get output

def translate(self):
		s_token = self.tokenizer_target.token_to_id("<S>")
		e_token = self.tokenizer_target.token_to_id("<E>")

		with torch.no_grad():
			self.model.eval()
			repeats = 3
			counter = 0
			texts_source_lang = []
			texts_target_lang = []
			texts_predictions = []

			for batch in self.validation_dataloader:
				if counter >= repeats: break
				counter += 1

				to_encoder = batch['to_encoder'].to(self.device)
				#to_decoder = batch['to_decoder'].to(self.device)
				mask_encoder = batch['mask_encoder'].to(self.device)
				#mask_decoder = batch['mask_decoder'].to(self.device)
				label = batch['label'].to(self.device)

				text_source = batch['text_source']

				if to_encoder.size(0) > 1: raise ValueError("For evaluation dimension must be 1!")

				# decode
				encoded = self.model.encode(to_encoder, mask_encoder)
				to_decoder = torch.empty(1,1).fill_(s_token).type_as(to_encoder).to(self.device)

				for iteration in range(0, self.max_tokens):
					mask_decoder = TranslationDataset.triangular_mask(to_decoder.size(1)).type_as(mask_encoder).to(self.device)
					
					# get output
					output = self.model.decode(encoded, mask_encoder, to_decoder, mask_decoder)

					p = self.model.project(output[:, -1])
					_, most_likely = torch.max(p, dim=1)

					if most_likely == e_token: break # we reached the end
					
					# next run with old content to decode + most likely token
					to_decoder = torch.cat(
						[
							to_decoder, #last input
							torch.empty(1,1).type_as(to_encoder).fill_(most_likely.item()).to(self.device)
						], dim=1
					)
				
				# get the sentences back out from the tokens
				estimated = self.tokenizer_target.decode(to_decoder.squeeze(0).detach().cpu().numpy())

				# add to lists
				texts_source_lang.append(text_source)
				texts_predictions.append(estimated)

				# print for debug
				print(f"Source: {text_source}")
				print(f"Predict: {estimated}")
			#raise ValueError("AAAAA")



#show output





