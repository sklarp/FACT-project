from PIL import Image
import torch
import os
from torchvision import transforms
from transformers import OFATokenizer, OFAModel, OFAConfig
# from generate import sequence_generator
from transformers.models.ofa.generate import sequence_generator
# from OFA.fairseq.fairseq import sequence_generator


mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

ckpt_dir = '/home/scur1045/FACT-project/OFA/OFA-HF-large-model'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
# tokenizer = OFATokenizer(vocab_file=f'{ckpt_dir}/vocab.json', merges_file=f'{ckpt_dir}/merges.txt')


txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids
# path_to_image = '/home/scur1045/FACT-project/HVV_EXPGEN_DATASET/Train_Val_Images/covid_memes_2076.png'
path_to_image = '/home/scur1045/FACT-project/HVV_EXPGEN_DATASET/Train_Val_Images/memes_1452.png'
# path_to_image = '/home/scur1045/FACT-project/img_test.png'
img = Image.open(path_to_image)
patch_img = patch_resize_transform(img).unsqueeze(0)


model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
generator = sequence_generator.SequenceGenerator(
                    tokenizer=tokenizer,
                    beam_size=5,
                    max_len_b=16, 
                    min_len=0,
                    no_repeat_ngram_size=3,
                )

data = {}
data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
gen_output = generator.generate([model], data)
gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

# using the generator of huggingface version
model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 

print(tokenizer.batch_decode(gen, skip_special_tokens=True))


