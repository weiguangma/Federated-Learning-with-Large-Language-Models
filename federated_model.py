#!/usr/bin/env python3
# /********************************************************************************
#  * Split Learning核心思想：
#  * 1. 客户端只保留embedding层，进行前向传播获取embedding
#  * 2. 将embedding传输到服务端进行后续计算和梯度更新
#  * 3. 服务端更新embedding权重后分发给所有客户端
#  * 4. 客户端不进行反向传播，只负责embedding计算
#  ********************************************************************************/

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Tokenizer
from typing import Dict, List, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenClientEmbedding(nn.Module):
    """
    Split Learning客户端嵌入层
    - 只负责前向传播：文本 -> embedding
    - 不进行反向传播，权重由服务端更新后同步
    """

    def __init__(self,
                 client_type: str = "general",
                 model_path: str = "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
                 max_length: int = 1024,
                 device: str = 'cuda'):
        super().__init__()

        self.client_type = client_type
        self.model_path = model_path
        self.max_length = max_length
        self.device = device

        logger.info(f"🔧 初始化 {client_type} 客户端嵌入层...")

        # 加载tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 加载完整模型以提取embedding层
        full_model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 确保数据类型一致
            device_map='cpu'
        )

        # 只保留embedding层
        self.embed_tokens = nn.Embedding(
            full_model.model.embed_tokens.num_embeddings,
            full_model.model.embed_tokens.embedding_dim,
            padding_idx=full_model.model.embed_tokens.padding_idx
        )

        # 复制权重
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                full_model.model.embed_tokens.weight)

        # Split Learning: 客户端不需要梯度计算
        self.embed_tokens.requires_grad_(False)

        # 清理完整模型以节省内存
        del full_model
        torch.cuda.empty_cache()

        logger.info(f"✅ {client_type}客户端嵌入层初始化完成:")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   嵌入维度: {self.embed_tokens.embedding_dim}")
        logger.info(f"   词汇表大小: {self.embed_tokens.num_embeddings}")
        logger.info(f"   最大长度: {max_length}")

    def update_embedding_weights(self, new_weights: torch.Tensor):
        """
        接收服务端更新的embedding权重
        这是Split Learning的核心：服务端训练后分发权重给客户端

        Args:
            new_weights: 服务端更新后的embedding权重
        """
        with torch.no_grad():
            self.embed_tokens.weight.copy_(new_weights)
        logger.debug(f"🔄 {self.client_type} 客户端embedding权重已更新")

    def forward(self, input_text: str) -> torch.Tensor:
        """
        Split Learning客户端前向传播：文本 -> embedding
        客户端只计算embedding，不进行后续处理

        Args:
            input_text: 输入文本

        Returns:
            embeddings: 嵌入向量 [seq_len, embedding_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors='pt',
        )

        # # Debug: 检查tokenization后的结果
        # decoded_text = self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=False)
        # logger.debug(f"🔍 解码后文本: {decoded_text}")

        input_ids = encoded['input_ids'].to(self.device)

        # Split Learning: 客户端只进行embedding查找，不计算梯度
        with torch.no_grad():
            # [1, seq_len, embedding_dim]
            embeddings = self.embed_tokens(input_ids)

        return embeddings.squeeze(0)  # [seq_len, embedding_dim]

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = device
        return self


class QwenServerModel(nn.Module):
    """Qwen服务端模型 - 包含完整的Qwen结构"""

    def __init__(self,
                 model_path: str = "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
                 device: str = 'cuda'):
        super().__init__()

        self.model_path = model_path
        self.device = device

        logger.info("🔧 初始化服务端Qwen2.5模型...")

        # 加载完整的Qwen模型
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 确保数据类型一致
            device_map='cpu'
        )

        # 加载tokenizer
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 获取分隔符token IDs
        self.sep_start_token = self.tokenizer.encode(
            "<|object_ref_start|>", add_special_tokens=False)[0]
        self.sep_end_token = self.tokenizer.encode(
            "<|object_ref_end|>", add_special_tokens=False)[0]

        logger.info("✅ 服务端Qwen2.5模型初始化完成:")
        logger.info(f"   模型路径: {model_path}")
        logger.info(f"   嵌入维度: {self.model.model.embed_tokens.embedding_dim}")
        logger.info(
            f"   词汇表大小: {self.model.model.embed_tokens.num_embeddings}")
        logger.info(
            f"   分隔符tokens: {self.sep_start_token}, {self.sep_end_token}")

    def create_federated_input_embeddings(self,
                                          server_instruction: str,
                                          client_embeddings: Dict[str, torch.Tensor],
                                          target_output: str = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], Optional[torch.Tensor]]:
        """
        Split Learning: 创建联邦输入embeddings (手动拼接Qwen格式)

        Args:
            server_instruction: 服务端指令
            client_embeddings: 客户端embeddings字典 {client_name: embeddings_tensor}
            target_output: 目标输出文本（训练时需要拼接到输入后面）

        Returns:
            combined_embeddings: 拼接后的embeddings [1, total_seq_len, embed_dim]
            combined_attention_mask: 拼接后的attention mask [1, total_seq_len]
            output_start_pos: output部分开始的位置（用于SFT损失计算），如果没有target_output则为None
            labels: 对应的token IDs标签 [1, total_seq_len]，如果没有target_output则为None
        """
        # 1. 获取特殊tokens的IDs
        im_start_token = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        all_embeddings = []
        all_masks = []
        all_token_ids = []  # 同时构建token IDs用于labels

        # 2. 添加服务端指令部分: <|im_start|>server\n{instruction}<|im_end|>
        # <|im_start|>
        im_start_ids = torch.tensor([[im_start_token]], device=self.device)
        im_start_emb = self.model.model.embed_tokens(im_start_ids)
        all_embeddings.append(im_start_emb)
        all_masks.append(torch.ones((1, 1), device=self.device))
        all_token_ids.append(im_start_token)

        # server\n
        server_role_text = "server\n" + server_instruction
        server_encoded = self.tokenizer(
            server_role_text,
            add_special_tokens=False,
            return_tensors='pt'
        )
        server_ids = server_encoded['input_ids'].to(self.device)
        server_emb = self.model.model.embed_tokens(server_ids)
        server_mask = server_encoded['attention_mask'].to(self.device)
        all_embeddings.append(server_emb)
        all_masks.append(server_mask)
        all_token_ids.extend(server_ids[0].tolist())

        # <|im_end|>
        im_end_ids = torch.tensor([[im_end_token]], device=self.device)
        im_end_emb = self.model.model.embed_tokens(im_end_ids)
        all_embeddings.append(im_end_emb)
        all_masks.append(torch.ones((1, 1), device=self.device))
        all_token_ids.append(im_end_token)

        # 3. 添加每个客户端的embedding: <|im_start|>client_name\n{client_embedding}<|im_end|>
        for client_name in sorted(client_embeddings.keys()):
            client_emb = client_embeddings[client_name]  # [seq_len, embed_dim]

            # <|im_start|>
            all_embeddings.append(im_start_emb)
            all_masks.append(torch.ones((1, 1), device=self.device))
            all_token_ids.append(im_start_token)

            # client_name\n
            client_role_text = f"{client_name}\n"
            client_role_encoded = self.tokenizer(
                client_role_text,
                add_special_tokens=False,
                return_tensors='pt'
            )
            client_role_ids = client_role_encoded['input_ids'].to(self.device)
            client_role_emb = self.model.model.embed_tokens(client_role_ids)
            client_role_mask = client_role_encoded['attention_mask'].to(
                self.device)
            all_embeddings.append(client_role_emb)
            all_masks.append(client_role_mask)
            all_token_ids.extend(client_role_ids[0].tolist())

            # 客户端embedding（添加batch维度）
            client_emb_batched = client_emb.unsqueeze(
                0)  # [1, seq_len, embed_dim]
            client_mask = torch.ones(
                (1, client_emb.size(0)), device=self.device)
            all_embeddings.append(client_emb_batched)
            all_masks.append(client_mask)
            # 注意：客户端内容的token IDs我们无法重建，用占位符（这部分不计算损失）
            all_token_ids.extend([0] * client_emb.size(0))

            # <|im_end|>
            all_embeddings.append(im_end_emb)
            all_masks.append(torch.ones((1, 1), device=self.device))
            all_token_ids.append(im_end_token)

        # 4. 如果有target_output，添加assistant部分
        output_start_pos = None
        if target_output is not None:
            # <|im_start|>
            all_embeddings.append(im_start_emb)
            all_masks.append(torch.ones((1, 1), device=self.device))
            all_token_ids.append(im_start_token)

            # assistant\n (角色标识符，不计算损失)
            assistant_role_text = "assistant\n"
            assistant_role_encoded = self.tokenizer(
                assistant_role_text,
                add_special_tokens=False,
                return_tensors='pt'
            )
            assistant_role_ids = assistant_role_encoded['input_ids'].to(
                self.device)
            assistant_role_emb = self.model.model.embed_tokens(
                assistant_role_ids)
            assistant_role_mask = assistant_role_encoded['attention_mask'].to(
                self.device)
            all_embeddings.append(assistant_role_emb)
            all_masks.append(assistant_role_mask)
            all_token_ids.extend(assistant_role_ids[0].tolist())

            # 记录实际输出内容开始位置（排除<|im_start|>assistant\n部分）
            output_start_pos = len(all_token_ids)

            # 检查是否是推理模式
            if target_output == "INFERENCE_MODE":
                # 推理模式：只添加<|im_start|>assistant\n，不添加具体内容和<|im_end|>
                # 模型将从这里开始生成
                pass  # 不添加任何内容，让模型自己生成
            else:
                # 训练模式：添加完整的target_output内容
                # {target_output} (实际输出内容，需要计算损失)
                output_encoded = self.tokenizer(
                    target_output,
                    add_special_tokens=False,
                    return_tensors='pt'
                )
                output_ids = output_encoded['input_ids'].to(self.device)
                output_emb = self.model.model.embed_tokens(output_ids)
                output_mask = output_encoded['attention_mask'].to(self.device)
                all_embeddings.append(output_emb)
                all_masks.append(output_mask)
                all_token_ids.extend(output_ids[0].tolist())

                # <|im_end|>
                all_embeddings.append(im_end_emb)
                all_masks.append(torch.ones((1, 1), device=self.device))
                all_token_ids.append(im_end_token)

            # logger.debug(f"🔍 Output开始位置: {output_start_pos}")

        # 5. 拼接所有embeddings
        # [1, total_seq_len, embed_dim]
        combined_embeddings = torch.cat(all_embeddings, dim=1)
        combined_attention_mask = torch.cat(
            all_masks, dim=1)   # [1, total_seq_len]

        # logger.debug(f"🔍 拼接后embeddings形状: {combined_embeddings.shape}")

        # 6. 根据模式进行不同的处理
        max_length = getattr(self, 'max_length', 1024)
        seq_len = combined_embeddings.size(1)

        # 判断是否是推理模式
        is_inference_mode = (target_output == "INFERENCE_MODE")

        if is_inference_mode:
            # 推理模式：不需要padding，只需要截断过长序列
            if seq_len > max_length:
                logger.warning(
                    f"⚠️ 推理序列长度 {seq_len} 超过max_length {max_length}，进行截断")
                combined_embeddings = combined_embeddings[:, :max_length, :]
                combined_attention_mask = combined_attention_mask[:, :max_length]
            # 推理模式不需要labels
            labels = None
        else:
            # 训练模式：需要构建labels并进行padding以保证批处理一致性
            # 确保token_ids长度与embeddings一致
            if len(all_token_ids) != seq_len:
                logger.warning(
                    f"⚠️ Token IDs长度 ({len(all_token_ids)}) 与embeddings长度 ({seq_len}) 不匹配")
                if len(all_token_ids) > seq_len:
                    all_token_ids = all_token_ids[:seq_len]
                else:
                    all_token_ids.extend([0] * (seq_len - len(all_token_ids)))

            if seq_len > max_length:
                # 截断：同步处理embeddings, attention_mask, token_ids, output_start_pos
                combined_embeddings = combined_embeddings[:, :max_length, :]
                combined_attention_mask = combined_attention_mask[:, :max_length]
                all_token_ids = all_token_ids[:max_length]

                # 如果output_start_pos被截断了，需要调整
                if output_start_pos is not None and output_start_pos >= max_length:
                    logger.warning(
                        f"⚠️ Output开始位置 {output_start_pos} 超过max_length {max_length}，调整为None")
                    output_start_pos = None

            elif seq_len < max_length:
                # 训练模式需要Padding：同步处理embeddings, attention_mask, token_ids
                pad_length = max_length - seq_len
                embed_dim = combined_embeddings.size(2)

                # Padding embeddings (用0填充)
                pad_embeddings = torch.zeros(
                    (1, pad_length, embed_dim), device=self.device)
                combined_embeddings = torch.cat(
                    [combined_embeddings, pad_embeddings], dim=1)

                # Padding attention mask (用0填充)
                pad_mask = torch.zeros((1, pad_length), device=self.device)
                combined_attention_mask = torch.cat(
                    [combined_attention_mask, pad_mask], dim=1)

                # Padding token IDs (用0填充)
                all_token_ids.extend([0] * pad_length)

            # 构建labels tensor (仅训练模式)
            labels = torch.tensor([all_token_ids], device=self.device)
            assert labels.size(1) == combined_embeddings.size(
                1), f"Labels长度 {labels.size(1)} 与embeddings长度 {combined_embeddings.size(1)} 不匹配"

        # logger.debug(f"🔍 最终embeddings形状: {combined_embeddings.shape}")
        # if output_start_pos is not None:
        #     logger.debug(f"🔍 SFT模式: 只计算位置 {output_start_pos} 之后的损失")

        return combined_embeddings, combined_attention_mask, output_start_pos, labels

    def forward_with_embeddings(self,
                                embeddings: torch.Tensor,
                                attention_mask: torch.Tensor,
                                labels: Optional[torch.Tensor] = None,
                                output_start_pos: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        使用预计算的embeddings进行前向传播

        Args:
            embeddings: 输入embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 标签token IDs [batch_size, seq_len]
            output_start_pos: output部分开始的位置（用于SFT只计算output部分的loss）

        Returns:
            输出字典包含logits和可选的loss
        """
        batch_size, seq_len = embeddings.shape[:2]
        position_ids = torch.arange(
            seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        # 直接使用Qwen2Model的transformer层
        outputs = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states)

        outputs = {"logits": logits}

        # SFT损失计算：只计算output部分的损失
        if labels is not None:
            # [batch_size, seq_len-1, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            # [batch_size, seq_len-1]
            shift_labels = labels[..., 1:].contiguous()

            # 如果指定了output_start_pos，只计算output部分的损失
            if output_start_pos is not None:
                # 注意：由于shift操作，output_start_pos需要减1
                # 原因：shift_labels去掉了第一个token，所以所有位置都向前移动了1
                adjusted_output_start_pos = max(0, output_start_pos - 1)

                # 创建loss mask：只对output部分计算损失
                loss_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                # 只有output部分为True
                loss_mask[:, adjusted_output_start_pos:] = True

                # 只选择output部分的logits和labels
                # [num_output_tokens, vocab_size]
                masked_logits = shift_logits[loss_mask]
                masked_labels = shift_labels[loss_mask]  # [num_output_tokens]

                if masked_logits.numel() > 0:  # 确保有输出tokens
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(masked_logits, masked_labels)
                else:
                    loss = torch.tensor(
                        0.0, device=self.device, requires_grad=True)

                # logger.debug(f"🔍 SFT损失计算: 原始output_start_pos={output_start_pos}, 调整后={adjusted_output_start_pos}, 输出tokens数量={masked_logits.size(0) if masked_logits.numel() > 0 else 0}")
            else:
                # 标准的全序列损失计算
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # logger.debug("🔍 使用全序列损失计算")

            outputs["loss"] = loss

        return outputs

    def generate_with_embeddings(self,
                                 embeddings: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 max_new_tokens: int = 100,
                                 temperature: float = 0.7,
                                 do_sample: bool = True) -> str:
        """
        使用embeddings生成文本 - 改进版本

        注意：由于Hugging Face的generate方法不支持inputs_embeds，
        我们需要手动实现generation loop，但这比较低效。
        更好的方法是将embeddings转回input_ids，但在Split Learning中这不可行。
        """
        try:
            with torch.no_grad():
                batch_size, seq_len = embeddings.shape[:2]
                # logger.debug(f"🔍 生成调试: 初始embeddings形状={embeddings.shape}")

                generated_tokens = []
                current_embeddings = embeddings.clone()
                current_mask = attention_mask.clone()

                # 获取初始序列长度，用于position_ids计算
                initial_seq_len = seq_len

                for step in range(max_new_tokens):
                    # 计算正确的position_ids
                    current_seq_len = current_embeddings.size(1)
                    position_ids = torch.arange(
                        current_seq_len, device=self.device).unsqueeze(0)

                    # 直接使用Qwen模型进行前向传播（避免重复的position_ids计算）
                    transformer_outputs = self.model.model(
                        inputs_embeds=current_embeddings,
                        attention_mask=current_mask,
                        position_ids=position_ids,
                        return_dict=True
                    )

                    hidden_states = transformer_outputs.last_hidden_state
                    logits = self.model.lm_head(hidden_states)

                    # 获取最后一个位置的logits
                    last_logits = logits[0, -1, :]  # [vocab_size]

                    # 采样下一个token
                    if do_sample and temperature > 0:
                        last_logits = last_logits / temperature
                        probs = torch.softmax(last_logits, dim=-1)
                        next_token_id = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token_id = torch.argmax(
                            last_logits, dim=-1, keepdim=True)

                    token_id = next_token_id.item()
                    generated_tokens.append(token_id)

                    # 检查结束条件
                    if token_id == self.tokenizer.eos_token_id:
                        # logger.debug(f"🔍 遇到EOS token，停止生成")
                        break

                    # 添加新token的embedding
                    next_token_embedding = self.model.model.embed_tokens(
                        next_token_id.unsqueeze(0))
                    current_embeddings = torch.cat(
                        [current_embeddings, next_token_embedding], dim=1)
                    current_mask = torch.cat(
                        [current_mask, torch.ones((1, 1), device=self.device)], dim=1)

                    # 防止序列过长
                    if current_embeddings.size(1) > 4096:
                        logger.warning("序列长度超过4096，停止生成")
                        break

                # 调试信息
                # logger.debug(f"🔍 生成完成: 共生成{len(generated_tokens)}个tokens")
                # logger.debug(f"🔍 生成的token ids: {generated_tokens[:10]}...")  # 只显示前10个

                # 解码生成的tokens
                if generated_tokens:
                    # 先尝试不跳过特殊token的解码
                    generated_text = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=False)
                    # logger.debug(f"🔍 解码结果(含特殊token)长度: {len(generated_text)}, 内容: '{generated_text[:100]}'")

                    # 如果包含特殊token，再尝试跳过特殊token的解码
                    generated_text_clean = self.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True)
                    # logger.debug(f"🔍 解码结果(跳过特殊token)长度: {len(generated_text_clean)}, 内容: '{generated_text_clean[:100]}'")

                    # 如果跳过特殊token后为空，返回原始解码结果
                    if generated_text_clean.strip():
                        return generated_text_clean.strip()
                    else:
                        # 移除常见的特殊token，但保留有意义的文本
                        filtered_text = generated_text.replace('<|endoftext|>', '').replace(
                            '<|im_start|>', '').replace('<|im_end|>', '')
                        # logger.debug(f"🔍 手动过滤后结果: '{filtered_text[:100]}'")
                        return filtered_text.strip()
                else:
                    # logger.debug(f"🔍 没有生成任何token")
                    return ""

        except Exception as e:
            logger.error(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            return "生成失败"

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = device
        return self


class FederatedQwenSystem(nn.Module):
    """
    Split Learning联邦Qwen系统

    架构设计：
    1. 客户端只保留embedding层，进行前向传播
    2. 服务端接收客户端embeddings，进行完整的前向和反向传播
    3. 服务端更新embedding权重后分发给所有客户端
    4. 客户端不参与反向传播，只负责embedding计算
    """

    def __init__(self,
                 model_path: str = "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
                 device: str = 'cuda',
                 enabled_clients: List[str] = None):
        super().__init__()

        self.model_path = model_path
        self.device = device

        # 默认启用所有客户端
        if enabled_clients is None:
            enabled_clients = ['port', 'railway', 'customs']
        self.enabled_clients = enabled_clients

        # 初始化启用的客户端embeddings
        self.client_embeddings = nn.ModuleDict()
        for client_name in enabled_clients:
            if client_name in ['port', 'railway', 'customs']:
                self.client_embeddings[client_name] = QwenClientEmbedding(
                    client_name, model_path, device=device
                )
            else:
                logger.warning(f"未知的客户端类型: {client_name}")

        logger.info(f"✅ 初始化联邦系统，启用客户端: {list(self.client_embeddings.keys())}")

        # 初始化服务端模型
        self.server_model = QwenServerModel(
            model_path=model_path, device=device)

        # 共享tokenizer
        self.tokenizer = self.server_model.tokenizer

        # 移动到指定设备
        self.to(device)

        # 🔄 关键：初始同步服务端embedding到客户端（在设备移动后）
        self._sync_embeddings_to_clients()

        logger.info("✅ 联邦Qwen系统初始化完成（已同步embedding）")

    def _sync_embeddings_to_clients(self):
        """
        🔄 Split Learning核心机制：将服务端embedding权重同步到所有客户端

        Split Learning的关键步骤：
        1. 服务端通过反向传播更新embedding权重
        2. 将更新后的权重分发给所有客户端
        3. 客户端接收新权重，用于下一轮前向传播
        4. 客户端不进行反向传播，只负责embedding计算
        """
        # logger.info("🔄 同步服务端embedding权重到客户端...")

        # 获取服务端的embedding权重 (这是ground truth)
        server_embed_weight = self.server_model.model.model.embed_tokens.weight.data

        # 同步到每个客户端
        sync_count = 0
        for client_name, client_embedding in self.client_embeddings.items():
            with torch.no_grad():
                # 直接复制服务端权重到客户端
                client_embedding.embed_tokens.weight.copy_(server_embed_weight)
            sync_count += 1
            # logger.debug(f"   ✅ {client_name}客户端embedding已同步")

        # logger.info(f"✅ 成功同步embedding权重到{sync_count}个客户端")

    def federated_step(self):
        """
        🔄 执行Split Learning的权重同步步骤

        Split Learning流程：
        1. 服务端已通过反向传播更新embedding权重
        2. 将更新后的权重分发给所有客户端
        3. 客户端接收新权重，准备下一轮前向传播

        注意：客户端不参与梯度计算，只接收权重更新
        这个方法应该在每个训练步骤的optimizer.step()之后调用
        """
        # Split Learning: 只需要同步权重，不需要聚合梯度
        # 因为客户端不计算梯度，所有梯度计算都在服务端完成
        self._sync_embeddings_to_clients()

        # logger.debug("🔄 Split Learning权重同步完成")

    def forward(self,
                server_instruction: str,
                client_instructions: Dict[str, str],
                labels: Optional[torch.Tensor] = None,
                target_output: str = None) -> Dict[str, torch.Tensor]:
        """
        Split Learning前向传播流程

        Decoder-only LLM的正确训练流程：
        1. 客户端各自计算embedding（不计算梯度）
        2. 服务端拼接：input + output（如果是训练）
        3. 服务端进行完整的前向传播
        4. 损失只在output部分计算

        Args:
            server_instruction: 服务端指令
            client_instructions: 客户端指令字典 {"port": "...", "railway": "..."}
            labels: 标签张量（用于计算损失）
            target_output: 目标输出文本（训练时必须提供）

        Returns:
            输出字典包含logits和loss
        """
        # 1. 客户端embedding编码 - Split Learning方式
        client_embeddings = {}
        for client_name, instruction in client_instructions.items():
            # 直接使用客户端名称，支持两种格式
            if client_name in self.client_embeddings:
                client_key = client_name
            elif '_' in client_name:
                # 处理 "client_01" 格式
                client_type = client_name.split('_')[1]
                if client_type == '01':
                    client_key = 'port'
                elif client_type == '02':
                    client_key = 'railway'
                elif client_type == '03':
                    client_key = 'customs'
                else:
                    client_key = 'port'  # 默认
            else:
                client_key = 'port'  # 默认

            # Split Learning: 客户端只进行前向传播
            embeddings = self.client_embeddings[client_key](instruction)
            client_embeddings[client_name] = embeddings

        # 2. 服务端拼接和处理（正确的decoder-only方式）
        combined_embeddings, combined_mask, output_start_pos, labels = self.server_model.create_federated_input_embeddings(
            server_instruction, client_embeddings, target_output
        )

        # 3. 服务端前向传播
        if self.training and target_output is not None:
            # SFT训练模式：使用已构建的labels和output_start_pos
            outputs = self.server_model.forward_with_embeddings(
                combined_embeddings, combined_mask, labels, output_start_pos
            )
        else:
            outputs = self.server_model.forward_with_embeddings(
                combined_embeddings, combined_mask, None
            )

        return outputs

    def generate(self,
                 server_instruction: str,
                 client_instructions: Dict[str, str],
                 max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 do_sample: bool = True) -> str:
        """生成文本"""
        # 客户端embedding编码
        client_embeddings = {}
        for client_name, instruction in client_instructions.items():
            # 根据客户端名称确定类型
            if 'port' in client_name.lower():
                client_key = 'port'
            elif 'railway' in client_name.lower():
                client_key = 'railway'
            elif 'customs' in client_name.lower():
                client_key = 'customs'
            else:
                # 如果名称不匹配，直接使用名称
                if client_name in self.client_embeddings:
                    client_key = client_name
                else:
                    client_key = 'port'  # 默认使用port

            embeddings = self.client_embeddings[client_key](instruction)
            client_embeddings[client_name] = embeddings

        # 服务端拼接（推理时需要添加assistant提示）
        combined_embeddings, combined_mask, _, _ = self.server_model.create_federated_input_embeddings(
            server_instruction, client_embeddings, target_output="INFERENCE_MODE"  # 特殊标识表示推理模式
        )

        # 生成
        return self.server_model.generate_with_embeddings(
            combined_embeddings, combined_mask, max_new_tokens, temperature, do_sample
        )

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = device
        for client_emb in self.client_embeddings.values():
            client_emb.to(device)
        self.server_model.to(device)
        return self


# 导出主要类
__all__ = ['QwenClientEmbedding', 'QwenServerModel', 'FederatedQwenSystem']
