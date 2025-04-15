import os
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from models.resnet import blip_resnet
from models.cvt import blip_cvt

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from transformers import AutoModel, AutoTokenizer


CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

class BLIP_Decoder(nn.Module):
    '''最终模型结构'''
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 prompt = '',
                 ):
        super().__init__()
        
        if args.image_encoder == 'cvt':
            args.image_size = 384
            vision_width = 768
            self.visual_encoder = blip_cvt(args)
        elif args.image_encoder == 'resnet101':
            args.image_size = 224
            vision_width = 2048
            self.visual_encoder = blip_resnet(args)
            
        self.args = args
        
        
        self.cls_head = nn.Linear(vision_width+512, 18*4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.vision_proj = nn.Linear(vision_width, 512)

        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1) # meomry是干啥的
        
        # define CXR-BERT
        url='BiomedVLP-CXR-BERT-specialized'
        self.encode_tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(
            url, trust_remote_code=True)
        # freeze the CXR-BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.img_emb_projection = nn.Linear(vision_width, 128)
        
    def forward(self, image, prompt, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        image_embeds, avg_embeds_ori = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        ##########################
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2)) # 21,16,512  
        query_embed = self.vision_proj(avg_embeds_ori) # 16,512
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None) # 1,16,1,512
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds_ori, hs), 1) # 16 2560

        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18) # 变成4分类问题了
        # logit adjustment
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        loss_cls = criterion_cls(cls_preds, cls_labels) # [4,18] 与 [18]
        
        prompts_captions = []
        for j in range(image_embeds.size(0)): # 长度为18
            prompts_captions.append(prompt[j] + caption[j])
        
        text = self.tokenizer(prompts_captions, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        text.input_ids[:,0] = self.tokenizer.bos_token_id # 把第一位赋值为bos，最后一位不用管

        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100 # 前N个不解码

        # 给出了labels 
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss     
        
        # 增加report 对比学习的loss
        tokenized_data = self.encode_tokenizer.batch_encode_plus(
            caption, add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors="pt").to(image.device)
        input_ids = tokenized_data.input_ids
        attention_mask = tokenized_data.attention_mask
        text_embs = self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids, attention_mask=attention_mask)
        
        # TODO: soft contrastive loss
        img_embs = self.img_emb_projection(avg_embeds_ori)
        cont_loss = self.infonce_loss(img_embs, text_embs, 0.07)
                   
        return loss_lm, loss_cls, cont_loss
        
    def generate(self, image, clip_memory, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds, avg_embeds = self.visual_encoder(image) # 16,49,2048; 16,2560
        
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2)) # B,clip_k,512 -> clip_k,B,512 
        query_embed = self.vision_proj(avg_embeds)
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)

        # classification branch
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist()

        prompts = []
        for j in range(len(cls_preds)): # 长度为18
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts.append(prompt)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)# 都是1
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        

        text = self.tokenizer(prompts, return_tensors="pt") # 输入只有prompt,没有其他 
        input_ids = text.input_ids.to(image.device) # 101为开头，102为结尾
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] # 为何去掉一维，去掉了102结尾
        attn_masks = attn_masks[:, :-1] 
        
        #beam search，没有给labels吗？无语
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                             min_length=min_length, # 4.25 Transformers
                                             max_new_tokens=max_length,
                                             num_beams=num_beams,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id, 
                                             repetition_penalty=repetition_penalty,
                                             attention_mask = attn_masks,
                                             **model_kwargs)            
            
        captions = []    
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(prompts[i]):])
        return captions, cls_preds, cls_preds_logits
    
    @staticmethod
    def infonce_loss(out_1, out_2, softmax_temperature):
        batch_size = out_1.size(0)
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        sim = out_2.detach() @ out_2.detach().t()
        lambda_ = 1.
        targets = lambda_ * \
            torch.eye(batch_size).type_as(sim) + (1 - lambda_) * sim

        logits = out_1 @ out_2.t()
        loss0 = F.cross_entropy(logits / softmax_temperature, targets)
        loss1 = F.cross_entropy(logits.t() / softmax_temperature, targets)
        cont_loss = (loss0 + loss1) / 2.

        return cont_loss

def blip_decoder(args, tokenizer, **kwargs):
    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model    
    
