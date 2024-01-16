from transformers import CLIPModel, CLIPConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn
import torch
import pdb

class Modified_ClipModel(CLIPModel):
    def __init__(self, config: CLIPConfig):
        super(Modified_ClipModel, self).__init__(config)

    def encode_text_feature(self, hidden_states, input_ids, attention_mask=None):
        output_attentions = self.text_model.config.output_attentions
        output_hidden_states = (
            self.text_model.config.output_hidden_states
        )
        return_dict = self.text_model.config.use_return_dict

        # hidden_states = self.text_encoder.text_model.embeddings(inputs_embeds=prompt_embeddings)
        hidden_states = hidden_states[:, :input_ids.argmax(-1)+1]
        input_ids = input_ids[:, :input_ids.argmax(-1)+1]
        position_ids = self.text_model.embeddings.position_ids[:, :input_ids.argmax(-1)+1]
        hidden_states = hidden_states + self.text_model.embeddings.position_embedding(position_ids)

        bsz, seq_len = input_ids.size()
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.text_model._build_causal_attention_mask(
                    bsz, seq_len, hidden_states.dtype).to(hidden_states.device)

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.text_model.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1)
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_text_feature_by_embedding(self, hidden_states, input_ids):
        text_outputs = self.encode_text_feature(hidden_states, input_ids)
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def forward_text_embedding(self, embeddings, ids, image_features,
                               object_mask=None,
                               ori_feature=None):
        text_features = self.get_text_feature_by_embedding(embeddings, ids)
        mse = torch.nn.MSELoss(reduction="sum").to(embeddings.device)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if object_mask is not None:  # keep the ori object feature
            assert ori_feature is not None, "style task must input original prompt feature when computing the object loss"
            mse_l = mse(text_features * (1 - object_mask), ori_feature * (1 - object_mask))
            # text_features.register_hook(lambda grad: grad * mask.float())
        else:
            mse_l = mse(image_features.mean(dim=0, keepdim=True), text_features)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, mse_l


