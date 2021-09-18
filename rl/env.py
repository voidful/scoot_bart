import random
from collections import defaultdict

import gym
import torch
import numpy
import math
import time

from transformers.models.bart.modeling_bart import _expand_mask


class Env(gym.Env):
    ACTION_MEANING = {
        0: "PASS",
        1: "EXIT_WITH_1_TOKEN"
    }

    def __init__(self, model, tokenizer, observation_input=[]):
        self.action_space = gym.spaces.Discrete(len(self.ACTION_MEANING))
        self.actions = list(self.ACTION_MEANING.keys())
        self.action_size = len(self.actions)
        self.model = model
        self.model.eval()
        self.encoder = model.get_encoder()
        self.decoder = model.get_decoder()
        self.tokenizer = tokenizer
        self.observation_space = observation_input
        self.reset()

    def reset(self, input_text=None, target_text=None):
        self.input_text = ""
        self.target_text = ""
        self.encoder_hidden = None
        self.decoder_hidden = None
        self.encoder_attention_mask = None
        self.decoder_attention_mask = None
        self.predicted = []
        self.act = []
        self.layer = 0
        self.num_step = 0
        self.past_key_values = defaultdict(list)
        self.finish = False
        if input_text is None:
            input_pair = random.choice(self.observation_space)
            self.input_text = input_pair[0]
            self.target_text = input_pair[1]
        else:
            self.input_text = input_text
            self.target_text = target_text
        if self.target_text and len(self.target_text) > 1:
            self.target_token = self.tokenizer.encode(self.target_text, add_special_tokens=False)
            self.target_step = len(self.target_token) * len(self.model.get_decoder().layers)
        return self._get_obs()

    def resize_past_keyvalue(self):
        for l in range(len(self.past_key_values)):
            past_key_value = []
            for i in range(2):
                past_key_value.append(self.past_key_values[l][i][:, :, :len(self.predicted), :])
            past_key_value.extend(self.past_key_values[l][2:])
            self.past_key_values[l] = tuple(past_key_value)

    def update_exit_past_keyvalue(self):
        for l in range(1, self.model.config.decoder_layers):
            past_key_value = []
            layer = l
            if len(self.past_key_values[l]) < 4 or self.past_key_values[l][0].shape[2] < len(self.predicted):
                layer = l - 1
            for i in range(2):
                past_key_value.append(self.past_key_values[layer][i][:, :, :len(self.predicted), :])
            past_key_value.extend(self.past_key_values[layer][2:])
            self.past_key_values[l] = tuple(past_key_value)

    def step(self, action, return_reward=True):
        self.num_step += 1
        if isinstance(action, numpy.ndarray):
            action = numpy.argmax(action)
        self.act.append(action)
        reward = 0
        if action == 0:
            if self.layer >= len(self.decoder.layers):
                action = 1

        if action > 0:
            # exit
            decode_head_hidden = self.model.lm_head(self.decoder_hidden).squeeze(0)
            decoded_num = action
            pred_tokens = torch.argmax(decode_head_hidden[: decoded_num, :], -1).tolist()

            self.resize_past_keyvalue()
            self.update_exit_past_keyvalue()

            if not self.finish and len(
                    self.predicted) < self.model.config.max_position_embeddings - self.action_size and self.tokenizer.eos_token_id not in pred_tokens:
                self.predicted.extend(pred_tokens)
            else:
                self.finish = True

            if return_reward:
                reward = self.get_reward()
            self.decoder_hidden = None
            self.layer = 0
        return self._get_obs(), reward, self.finish, {'predicted': self.predicted, 'step': self.num_step,
                                                      'action': self.act}

    def get_reward(self):
        reward = 0
        for i, t in enumerate(self.predicted):
            if i >= len(self.target_token):
                reward -= 1
            elif self.target_token[i] == t:
                reward += 1
            else:
                reward -= 1

        if self.finish:
            for i, t in enumerate(self.target_token):
                if i >= len(self.predicted):
                    reward -= 1
                elif self.predicted[i] == t:
                    reward += 1
                else:
                    reward -= 1
            reduce_step = self.target_step / self.num_step
            reward *= reduce_step

        return reward

    def _get_obs(self):
        with torch.no_grad():
            if self.encoder_hidden is None:
                feature_dict = self.tokenizer(self.input_text, return_tensors='pt').to(self.model.device)
                self.encoder_dict = feature_dict
                self.encoder_hidden = self.encoder(**feature_dict)[0]
                self.predicted = [self.tokenizer.eos_token_id]

            if self.decoder_hidden is None:
                extra_mask_tok = [self.tokenizer.mask_token_id] * (self.action_size - 2)
                pred_tok = [self.predicted[-1]] + extra_mask_tok
                pred_atten = self.predicted + extra_mask_tok
                input_ids = torch.tensor([pred_tok])
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1]).to(self.model.device)
                inputs_embeds = self.decoder.embed_tokens(input_ids)
                self.decoder_attention_mask = torch.ones(len(pred_atten)).unsqueeze(0)

                # past_key_values_length
                past_key_values_length = self.past_key_values[0][0].shape[2] if len(
                    self.past_key_values[0]) > 0 else 0
                if self.decoder_attention_mask.shape[1] >= past_key_values_length:
                    self.decoder_attention_mask = None
                self.decoder_attention_mask = self.decoder._prepare_decoder_attention_mask(self.decoder_attention_mask,
                                                                                           input_shape, inputs_embeds,
                                                                                           past_key_values_length)
                self.encoder_attention_mask = _expand_mask(self.encoder_dict['attention_mask'], inputs_embeds.dtype,
                                                           tgt_len=input_shape[-1])
                # embed positions
                positions = self.decoder.embed_positions(input_shape, past_key_values_length)
                self.decoder_hidden = inputs_embeds + positions
                self.decoder_hidden = self.decoder.layernorm_embedding(self.decoder_hidden)

            past_key_value = self.past_key_values[self.layer] if len(self.past_key_values[self.layer]) > 0 else None
            layer_outputs = self.decoder.layers[self.layer](
                self.decoder_hidden,
                attention_mask=self.decoder_attention_mask,
                encoder_hidden_states=self.encoder_hidden,
                encoder_attention_mask=self.encoder_attention_mask,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False
            )
            self.decoder_hidden = layer_outputs[0]
            self.past_key_values[self.layer] = layer_outputs[1]
            self.layer += 1
            return self.decoder_hidden.squeeze(0)[0]
