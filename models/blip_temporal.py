import os, sys, math
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

import numpy as np

from models.temporal_transformer import CascadedTemporalModule
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


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
        self.args = args
        
        if args.image_encoder == 'cvt':
            args.image_size = 384
            vision_width = 768
            self.visual_encoder = blip_cvt(args)
        elif args.image_encoder == 'resnet101':
            args.image_size = 224
            vision_width = 2048
            self.visual_encoder = blip_resnet(args)
        
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
        
        
        url='BiomedVLP-CXR-BERT-specialized'
        self.encode_tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(
            url, trust_remote_code=True)
        # freeze the CXR-BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.img_emb_projection = nn.Linear(vision_width, 128)
        
        group_size = 49
        self.min_seq_len = 2
        self.max_seq_len = 5
        num_heads = 4
        self.visual_temporal_model = CascadedTemporalModule(
            group_size=group_size,
            block_size=self.max_seq_len,
            n_embd=vision_width,
            num_heads=num_heads,
            embd_pdrop=0.1,
            n_layer=2
        )
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        # for param in self.memory.parameters():
        #     param.requires_grad = False
        # for param in self.cls_head.parameters():
        #     param.requires_grad = False       
        # for param in self.vision_proj.parameters():
        #     param.requires_grad = False
        # for param in self.text_decoder.parameters():
        #     param.requires_grad = False
        map_size = int(math.sqrt(49))
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)

    def forward(self, images, clip_memory, prompt, caption, report_comparision, cls_labels, batch_mask, study_date, criterion_cls, base_probs):
        B, T, C, H, W = images.shape
        batch_len = batch_mask.sum(dim=1)
        nonull_images = images[batch_mask] # 把B和length整合 ,B 3, 248,248的图像
        image_embs, image_embs_avg = self.visual_encoder(nonull_images)
        
        nonull_image_embs_list = torch.split(
            image_embs, batch_len.tolist(), dim=0)
        pad_image_embs = pad_sequence(
            nonull_image_embs_list, batch_first=True)
        pad_img_len = self.max_seq_len - pad_image_embs.shape[1]
        B, _, N, D = pad_image_embs.shape
        zero_image_embs = torch.zeros(
            B, pad_img_len, N, D).type_as(image_embs)
        image_embs = torch.cat(
            [pad_image_embs, zero_image_embs], dim=1) 

        temporal_image_embs = self.visual_temporal_model(
            image_embs, study_date)
        temporal_image_embs = rearrange(
            temporal_image_embs, "(b t) n d -> b t n d", b=B, t=T) # B, T, 49, 512
        temporal_nonull_img_embs = temporal_image_embs[batch_mask] # 41, 49,512
          
        prompts_captions = []
        captions_bt = []
        report_comparision_bt = []
        for j in range(B): # 长度为18
            for i in range(T): 
                prompts_captions.append(prompt[i][j] + caption[i][j])
                if len(caption[i][j])>0:
                    captions_bt.append(caption[i][j])
                if len(report_comparision[i][j])>0:
                    report_comparision_bt.append(report_comparision[i][j])
        
        text = self.tokenizer(prompts_captions, padding='longest', truncation=True, return_tensors="pt").to(images.device)
        text.input_ids[:,0] = self.tokenizer.bos_token_id # 把第一位赋值为bos，最后一位不用管
        
        input_ids = torch.stack(torch.chunk(text.input_ids, B, dim=0),dim=0) # B, T, 512
        attention_mask = torch.stack(torch.chunk(text.attention_mask, B, dim=0),dim=0) # B, T, 512
        
        batch_input_ids = rearrange(input_ids, "b t d -> (b t) d")
        batch_attention_mask = rearrange(attention_mask, "b t d -> (b t) d")
        batch_input_ids = batch_input_ids[batch_mask.reshape(-1)]
        batch_attention_mask = batch_attention_mask[batch_mask.reshape(-1)]
        
        decoder_targets = batch_input_ids.masked_fill(batch_input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100 # 前N个不解码

        # 给出了labels 
        decoder_output = self.text_decoder(batch_input_ids, 
                                           attention_mask = batch_attention_mask, 
                                           encoder_hidden_states = temporal_nonull_img_embs,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
          
        loss_lm = decoder_output.loss     
        
        ##########################
        # NxKxC -> KxNxC        
        clip_memory = clip_memory[batch_mask] 
        clip_memory = torch.permute(clip_memory, (1, 0, 2)) # 21,16,512  
        
        embeds_ori =  torch.permute(temporal_nonull_img_embs,(0,2,1)) # 41, 512, 49
        embeds_ori = embeds_ori.view(temporal_nonull_img_embs.size(0), temporal_nonull_img_embs.size(-1), 7, 7)
        avg_embeds_ori = self.avg_fnt(embeds_ori).flatten(1) # 41, 512
        query_embed = self.vision_proj(avg_embeds_ori) # 16,512
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None) # 1,16,1,512
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds_ori, hs), 1) # 16 2560

        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18) # 变成4分类问题了
        # logit adjustment
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(images.device)
        cls_labels = cls_labels[batch_mask]
        loss_cls = criterion_cls(cls_preds, cls_labels) # [4,18] 与 [18]
        
        # contrastive loss
        tokenized_data = self.encode_tokenizer.batch_encode_plus(
            report_comparision_bt, add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_tensors="pt").to(images.device)
        input_ids = tokenized_data.input_ids
        attention_mask = tokenized_data.attention_mask
        text_embs = self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids, attention_mask=attention_mask)
        img_embs = self.img_emb_projection(avg_embeds_ori)
        cont_loss = self.infonce_loss(img_embs, text_embs, 0.07)
                   
        return loss_lm, loss_cls, cont_loss
  
    def generate(self, images, clip_memory, batch_mask, study_date, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        B, T, C, H, W = images.shape
        study_date = study_date.to(images.device)
        batch_len = batch_mask.sum(dim=1)
        nonull_images = images[batch_mask] # 把B和length整合 ,B 3, 248,248的图像
        image_embs, _ = self.visual_encoder(nonull_images)
        
        # resume temporal array
        nonull_image_embs_list = torch.split(
            image_embs, batch_len.tolist(), dim=0)
        pad_image_embs = pad_sequence(
            nonull_image_embs_list, batch_first=True)
        pad_img_len = self.max_seq_len - pad_image_embs.shape[1]
        B, _, N, D = pad_image_embs.shape
        zero_image_embs = torch.zeros(
            B, pad_img_len, N, D).type_as(image_embs)
        image_embs = torch.cat(
            [pad_image_embs, zero_image_embs], dim=1)

        temporal_image_embs = self.visual_temporal_model(
            image_embs, study_date)
        temporal_image_embs = rearrange(
            temporal_image_embs, "(b t) n d -> b t n d", b=B, t=T)
        temporal_nonull_img_embs = temporal_image_embs[batch_mask]
        
        
        # NxKxC -> KxNxC
        clip_memory = clip_memory[batch_mask]        
        clip_memory = torch.permute(clip_memory, (1, 0, 2)) # B,clip_k,512 -> clip_k,B,512 
        embeds_ori =  torch.permute(temporal_nonull_img_embs,(0,2,1)) # 41, 512, 49
        embeds_ori = embeds_ori.view(temporal_nonull_img_embs.size(0), temporal_nonull_img_embs.size(-1), 7, 7)
        avg_embeds_ori = self.avg_fnt(embeds_ori).flatten(1) # 41, 512
        query_embed = self.vision_proj(avg_embeds_ori) # 16,512
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds_ori, hs), 1)

        # classification branch
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds_ori = torch.argmax(cls_preds, dim=1)        
        
        if not sample:
            temporal_nonull_img_embs = temporal_nonull_img_embs.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(temporal_nonull_img_embs.size()[:-1],dtype=torch.long).to(images.device)# 都是1
        model_kwargs = {"encoder_hidden_states": temporal_nonull_img_embs, "encoder_attention_mask":image_atts}
        
        
        prompts_only = []
        cls_preds = cls_preds_ori.cpu().numpy().tolist()
        for j in range(len(cls_preds)): # 长度为18
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts_only.append(prompt)

        text = self.tokenizer(prompts_only, return_tensors="pt") # 输入只有prompt,没有其他 
        input_ids = text.input_ids.to(images.device) # 101为开头，102为结尾
        attn_masks = text.attention_mask.to(images.device)
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
            captions.append(caption[len(prompts_only[i]):])
        return captions, cls_preds_ori.cpu().numpy().tolist(), cls_preds_logits
    
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
    
