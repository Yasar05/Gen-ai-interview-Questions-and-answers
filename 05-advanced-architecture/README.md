# Advanced Architecture - Generative AI Interview Questions

**Powered by [Euron](https://euron.one/) - Your AI Innovation Partner**

## Questions 51-65: Advanced Architecture Concepts

---

### 51. Explain the differences between GPT, LLaMA, and Claude architectures.

**Answer:**

All 3 models are decoder only transformers, with causal self attention and autoregression next token prediction.
Causal self-attention is a type of attention mechanism where each token is allowed to attend only to previous tokens and itself, but never to future tokens.
**GPT (Generative Pre-trained Transformer):**
Heavy RLHF layers, Built-in tool calling & function execution, Multimodal extensions (text, image, audio), System-message & role-based control deeply integrated
Optimized for API reliability at global scale. Design Goal: Maximum instruction-following, creativity, and production reliability
- **Architecture**: Decoder-only transformer
- **Training**: Autoregressive language modeling
- **Parameters**: 175B (GPT-3), ~1.7T (GPT-4)
- **Strengths**: Strong generation, creative writing
- **Use Cases**: Text generation, completion, creative tasks

**LLaMA (Large Language Model Meta AI):**
Extremely clean & minimal transformer design, No heavy built-in RLHF, No proprietary tool layers, Designed for:,LoRA / QLoRA, PEFT, On-prem deployment,Optimized for parameter efficiency. Open, efficient, fine-tune-friendly foundation model
- **Architecture**: Decoder-only transformer
- **Training**: Self-supervised learning
- **Parameters**: 7B, 13B, 30B, 65B
- **Strengths**: Open-source, efficient, research-friendly
- **Use Cases**: Research, fine-tuning, open development
only llama is open source. the rest others arent. GP and claude has limited finetuning
**Claude (Anthropic):**
Claude is trained using: Constitutional AI → The model learns safety rules from a written “constitution” instead of relying only on human feedback. Some RLHF, but much less than GPT. This makes Claude: More self-correcting, More consistent in refusing harmful requests, More aligned without massive human labeling
🔹 Design Goal: Maximum safety, harmlessness, and long-context reasoning
- **Architecture**: Transformer-based
- **Training**: Constitutional AI, RLHF
- **Parameters**: ~100B-1T
- **Strengths**: Safety, helpfulness, harmlessness
- **Use Cases**: Safe AI, helpful assistance, reasoning

**Key Differences:**
| Aspect | GPT | LLaMA | Claude |
|--------|-----|-------|--------|
| **Openness** | Closed | Open | Closed |
| **Safety** | Standard | Standard | High |
| **Efficiency** | Medium | High | Medium |
| **Research** | Limited | Full | Limited |

| Feature                 | **GPT**                         | **LLaMA**                  | **Claude**                         |
| ----------------------- | ------------------------------- | -------------------------- | ---------------------------------- |
| Base Architecture       | Decoder-only Transformer        | Decoder-only Transformer   | Decoder-only Transformer           |
| Attention Type          | Causal self-attention           | Causal self-attention      | Causal self-attention              |
| Core Training           | Autoregressive + SFT + RLHF     | Autoregressive pretraining | Autoregressive + Constitutional AI |
| Safety Handling         | RLHF-heavy                      | Mostly external            | Built-in deeply                    |
| Tool Calling            | Native                          | Custom-built by users      | Native                             |
| Open Source             | ❌ No                            | ✅ Yes                      | ❌ No                               |
| Fine-tuning Flexibility | ⚠️ Limited                      | ✅ Excellent                | ⚠️ Limited                         |
| Context Length          | Large                           | Medium–Large               | ✅ Very Large                       |
| Primary Optimization    | Product & instruction following | Efficiency & research      | Safety & reasoning                 |

---

### 52. What is Mixture of Experts (MoE) and how does it work?

**Answer:**

this is dense feedforward neural network wothout MOE. attention here is multi head self attention
Layer 1 → Attention + FFN
Layer 2 → Attention + FFN
Layer 3 → Attention + FFN
...
Layer N → Attention + FFN

if we use moe, in each layer instead of 1 ffn we will have multiple ffn
Self-Attention
↓
MULTIPLE FFNs (Experts)
Layer1 will have
FFN Expert 1
FFN Expert 2
FFN Expert 3
...
FFN Expert 16

Mixture of Experts is a neural network architecture where instead of using one big model for every input, the system contains many smaller “expert” networks, and only a few of them are activated for each input. In short: Not all neurons work every time — only the most relevant “experts” are used per token. This gives you:
✅ Massive model capacity ✅ Much lower compute per request.
Traditional dense models: Activate all parameters for every token. As models grow → cost grows linearly with size This becomes: ❌ Too expensive, ❌ Too slow
❌ Too energy-hungry. MoE solves this by: Having many experts. Activating only 1–4 experts per token. So you get: The power of a trillion-parameter model at the cost of a 50–100B model per request.
Attention = communication between tokens
FFN = thinking / transformation done on each token individually
**How MoE Works:**
Let’s say your MoE layer has: a. 64 experts b. A router (also called a gating network). Only 2 experts are activated per token
A token’s hidden vector reaches the MoE layer in the transformer.
Here’s what happens:
1. **Expert Networks**: Multiple specialized sub-networks
2. **Gating Network**: Routes inputs to relevant experts. A small neural network called the router looks at the token and decides: “Which experts are best suited for this token?” It outputs: A score for each expert. Chooses Top-K experts (e.g., top-2)
3. **Sparse Activation**: Only selected experts process input. Instead of all 64 experts running: Only the chosen 2 experts process the token. The remaining 62 do nothing
4. **Combination**: Weighted combination of expert outputs. The outputs of the selected experts are: Weighted, Combined, Passed to the next transformer layer
In most designs: MoE replaces the Feed-Forward Network (FFN) inside some transformer layers. Attention layers remain dense. Only FFN becomes expert-based
**Benefits:**
- **Efficiency**: Lower computational cost
- **Scalability**: Can handle larger models
- **Specialization**: Experts learn specific patterns
- **Performance**: Better results with fewer parameters

**Implementation:**
```python
class MoELayer:
    def __init__(self, num_experts, expert_size):
        self.experts = [Expert(expert_size) for _ in range(num_experts)]
        self.gate = GatingNetwork(num_experts)
    
    def forward(self, x):
        # Route to experts
        expert_weights = self.gate(x)
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            if expert_weights[i] > threshold:
                expert_outputs.append(expert(x) * expert_weights[i])
        
        return sum(expert_outputs)
```

---

### 53. Explain the concept of sparse attention patterns.

**Answer:**

Sparse attention patterns reduce computational complexity by limiting which positions each token can attend to, rather than attending to all positions. Sparse attention patterns mean that each token attends only to a selected subset of other tokens instead of attending to every token in the sequence. In simple words:
The model does not look at everything — it only looks at what matters. This is in contrast to dense (full) attention, where: Every token attends to all other tokens. Computational cost grows as O(n²)
In standard transformers: If you have 10,000 tokens → attention compares 10,000 × 10,000. That’s 100 million interactions. This causes:❌ High memory usage
❌ Very high latency ❌ GPU out-of-memory errors. Sparse attention solves this by:✅ Reducing computation ✅ Reducing memory ✅ Enabling long-context models
**Types of Sparse Attention:**
- **Local Attention**: Attend to nearby positions. Each token only attends to: A fixed window of nearby tokens. Example: Token 100 only attends to tokens 90–110
- **Strided Attention**: Attend to every k-th position. Tokens attend at fixed intervals: Example:Token attends to positions: 1, 10, 20, 30, …
- **Random Attention**: Random subset of positions. Each token attends to: A small random set of distant tokens. Maintain global connectivity. Avoid losing long-range information.
- **Block Attention**: Attend within blocks. The sequence is divided into blocks, and: Tokens attend only within: Their own block. Selected neighboring blocks

**Benefits:**
- **Efficiency**: O(n) instead of O(n²) complexity
- **Scalability**: Handle longer sequences
- **Performance**: Often maintains quality
- **Memory**: Lower memory requirements

Which Models Use Sparse Attention?
✅ Longformer → Local + Global
✅ BigBird → Local + Global + Random
✅ Reformer → Local + LSH-based sparse attention
✅ FlashAttention (optimized dense, not sparse pattern) → reduces memory but still dense
✅ Claude & GPT (partially optimized variants internally) for long context

**Example Patterns:**
```
Full Attention:    [1,1,1,1,1,1,1,1]
Local Attention:   [1,1,1,0,0,0,0,0]
Strided Attention: [1,0,1,0,1,0,1,0]
```

---

### 54. What is the difference between LoRA and full fine-tuning?

**Answer:**
Full fine-tuning updates all model parameters, making the entire model change. LoRA (Low-Rank Adaptation) updates only a tiny set of added parameters while keeping the original model frozen.
**LoRA (Low-Rank Adaptation):**
- **Method**: Freezes base model, adds trainable low-rank matrices. Inserts small trainable low-rank matrices into: Attention layers. Sometimes FFN layers
Only these small matrices are trained. So instead of updating billions of weights, you update: ✅ Only a few million parameters
- **Parameters**: Only ~1% of model parameters
- **Memory**: Much lower memory requirements. 10x–100x less GPU memory
- **Speed**: Faster training
- **Performance**: Often comparable to full fine-tuning

**Full Fine-tuning:**
- **Method**: Updates all model parameters
- **Parameters**: All model parameters
- **Memory**: High memory requirements
- **Speed**: Slower training
- **Performance**: Best possible performance

**Comparison:**
| Aspect | LoRA | Full Fine-tuning |
|--------|------|------------------|
| **Parameters** | ~1% | 100% |
| **Memory** | Low | High |
| **Speed** | Fast | Slow |
| **Performance** | Good | Best |
| **Flexibility** | Limited | High |

**When to Use:**
- **LoRA**: Limited resources, quick adaptation
- **Full Fine-tuning**: Maximum performance, sufficient resources

---

### 55. How does QLoRA work and what are its benefits?

**Answer:**

QLoRA (Quantized Low-Rank Adaptation)) combines quantization with LoRA to enable efficient fine-tuning of large models on consumer hardware.
QLoRA is a fine-tuning technique that combines: ✅ Quantization (to shrink the base model into low precision)✅ LoRA adapters (to train only small new parameters)
So in one line: QLoRA lets you fine-tune very large LLMs on a single GPU by training only small LoRA adapters on top of a quantized base model.
**How QLoRA Works:**
Let’s say you want to fine-tune a 65B LLaMA model on your local machine. The original LLM weights (normally 16-bit or 32-bit) are: Compressed down to 4-bit precision. Stored using special quantization tricks. 
1. **Quantization**: Reduce model precision (e.g., 4-bit). Quantization is the process of shrinking the numerical precision of the model’s weights. Normally, model weights are stored in: 32-bit (FP32) or 16-bit (FP16). QLoRA compresses them down to: ✅ 4-bit precision
2. **LoRA**: Add low-rank adaptation layers. Small LoRA matrices are added to: Attention projection layers Sometimes feed-forward layers. What it means:
Instead of changing the original model weights, QLoRA: Inserts small trainable LoRA matrices into: Attention projection layers. Sometimes feed-forward layers
These LoRA layers: Are full precision (16-bit) Are very small compared to the base model. Are the only weights that are trained. So: ✅ Base model = frozen & quantized ✅ LoRA adapters = trainable & precise
3. **Gradient Checkpointing**: Reduce memory usage. What it means: During training, neural networks normally store: All intermediate activations. For use in backpropagation. This consumes a huge amount of GPU memory. With gradient checkpointing: The model does not store all activations. It recomputes some of them during backpropagation instead. This trades: ✅ Less memory usage ❌ Slightly more compute time. But that’s a very good trade-off when GPUs are limited.
4. **Efficient Training**: Train on quantized model. What it means together: All three ideas combine to allow this: ✅ Base model is tiny (4-bit quantized)
✅ Only small LoRA adapters are trained ✅ Memory is further reduced using checkpointing. So now you can: Fine-tune very large models On a single GPU At low cost
Without losing much accuracy.This is why QLoRA was a breakthrough.
Without QLoRA: You need massive GPUs to fine-tune large models. With QLoRA: You compress the model, Freeze it, Add tiny trainable layers, Reduce memory during training, And still get near full fine-tuning performance 
**Benefits:**
- **Memory Efficiency**: 4x less memory than full fine-tuning
- **Speed**: Faster training
- **Accessibility**: Run on consumer GPUs
- **Performance**: Maintains model quality
- **Cost**: Lower computational costs

**Implementation:**
```python
# QLoRA setup
model = load_model_4bit(model_name)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
```

---

### 56. Explain the concept of model parallelism in large language models.

**Answer:**

Model parallelism distributes model parameters across multiple devices to handle models that don't fit on a single GPU. Model parallelism is a technique where the parameters of a large neural network are split across multiple GPUs or machines so that the model can be trained and run even when it is too large to fit into the memory of a single GPU. In simple words: Instead of splitting the data, we split the model itself. This is essential for training and serving very large models like GPT-3, GPT-4, LLaMA-70B+, and trillion-parameter models.
Modern LLMs contain: Tens or hundreds of billions of parameters Require hundreds of gigabytes of memory. But a single GPU typically has: 24GB, 40GB, or 80GB memory
So:A single GPU cannot hold the full model, making model parallelism mandatory.

**Types of Model Parallelism:**
- **Tensor Parallelism**: (Tensor Model Parallelism (Intra-Layer Parallelism)). Split tensors across devices. A single layer is split across multiple GPUs. Each GPU stores only a slice of the layer’s weight matrix. All GPUs compute in parallel. Results are combined using communication operations
This is used for: Very large attention layers, Very wide feed-forward networks
- **Pipeline Parallelism**: Split layers across devices. Different layers of the model are placed on different GPUs. GPU 1 → early layers. GPU 2 → middle layers
GPU 3 → later layers. Mini-batches flow through the GPUs like an assembly line, improving memory efficiency.
- **Data Parallelism**: Split data across devices.
- **Hybrid**: Combine multiple approaches.
- “Data parallelism splits the training data across GPUs while keeping a full copy of the model on each device, whereas hybrid parallelism combines data parallelism with tensor and pipeline model parallelism to scale both model size and training speed for extremely large language models.”
How Large Models Are Trained in Practice: In real LLM training, companies combine: Data Parallelism → same model on multiple GPUs, different data
Tensor Parallelism → split layers across GPUs, Pipeline Parallelism → split layers across stages
This is called: ✅ 3D Parallelism
Used in: GPT-3, GPT-4 (likely), LLaMA-70B+, DeepSeek, Mixtral, etc.
**Tensor Parallelism:**
```python
# Split attention across devices
class ParallelAttention:
    def __init__(self, num_devices):
        self.devices = num_devices
        self.attention_heads = split_heads(num_devices)
    
    def forward(self, x):
        # Distribute across devices
        results = []
        for i, device in enumerate(self.devices):
            results.append(self.attention_heads[i](x, device))
        return combine_results(results)
```

**Benefits:**
- **Scalability**: Handle larger models
- **Memory**: Distribute memory requirements
- **Performance**: Parallel computation
- **Flexibility**: Mix and match strategies

---

### 57. What is the purpose of gradient checkpointing?

**Answer:**

Gradient checkpointing reduces memory usage during training by recomputing activations instead of storing them, trading computation for memory.The purpose of gradient checkpointing is to reduce GPU memory usage during training by not storing all intermediate activations in memory and instead recomputing them during the backward pass when needed. In simple words:You trade extra computation for much lower memory usage.
During training, neural networks normally: Store all forward-pass activations, Use them for backpropagation, This consumes a huge amount of GPU memory, especially in: Transformers, Long sequence training, Very deep models
For large LLMs, this memory requirement often: ❌ Exceeds GPU limits ❌ Makes training impossible on available hardware. Gradient checkpointing solves this.
**How it Works:**
1. **Forward Pass**: Store only checkpoint activations. During the normal forward pass of a neural network, every layer produces intermediate values called activations, and by default, all of them are stored in GPU memory for use during backpropagation. With gradient checkpointing, the model does not store activations for every layer. Instead, it stores activations only at a few selected layers called checkpoints. All other intermediate activations are intentionally discarded to save memory. This is where the main memory reduction begins.
2. **Backward Pass**: Recompute intermediate activations. During the backward pass (backpropagation), the model needs the intermediate activations to compute gradients. Since most of them were not stored during the forward pass, the model recomputes those missing activations on the fly by re-running parts of the forward computation between two checkpoints. This allows correct gradient calculation without needing to store everything in memory.
3. **Memory Savings**: Significant memory reduction. Because the model only stores a small number of checkpoint activations instead of all layer activations, GPU memory usage drops dramatically
4. **Computation Trade-off**: More computation, less memory. The key trade-off of gradient checkpointing is: ✅ Much lower memory usage
❌ Slightly higher computation time. This happens because some parts of the forward pass are executed twice—once during the real forward pass and again during the backward pass for recomputation. In practice, this usually increases training time by about 20–40%, but the memory savings are often worth it.

**Benefits:**
- **Memory**: 50-80% memory reduction
- **Scalability**: Train larger models
- **Accessibility**: Run on smaller GPUs
- **Cost**: Lower hardware requirements

**Implementation:**
```python
import torch.utils.checkpoint as checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint.checkpoint(self.forward_pass, x)

def forward_pass(self, x):
    # Forward pass implementation
    return self.layers(x)
```

---

### 58. How does flash attention improve efficiency?

**Answer:**

Flash Attention is an efficient attention implementation that reduces memory usage and improves speed through better memory access patterns.
FlashAttention is an optimized way to compute the attention mechanism that dramatically reduces memory usage and increases speed by avoiding the explicit storage of large attention matrices. In simple words: It computes attention in small blocks directly on the GPU without ever materializing the full attention matrix in memory.

In standard attention mechanism, The problem is:QKt is an N × N matrix. Memory complexity = O(N²). For long sequences (8k, 32k, 100k tokens), this becomes:
❌ Too slow ❌ GPU out-of-memory. Even if the math is optimized, the memory read/write (I/O) becomes the real bottleneck. 
FlashAttention improves efficiency in four key ways:
**Key Innovations:**
- **Tiling**: Process attention in blocks. Tiling means FlashAttention splits the large attention computation into small blocks (tiles) of queries, keys, and values. Instead of computing attention for the entire sequence at once, it processes one small block at a time. These blocks are sized to fit inside the GPU’s fast on-chip memory (shared memory). This avoids loading huge matrices into slow GPU memory and makes computation much faster and more memory-efficient.
- **Recomputation**: Trade memory for computation. FlashAttention avoids storing large intermediate results like the full attention matrix. Instead, it recomputes some intermediate values during execution rather than saving them in memory. This means it performs a bit of extra computation, but in return it dramatically reduces memory usage. This memory–compute trade-off is what makes long-context attention feasible on real GPUs.
- **Memory Access**: Optimized memory patterns. In standard attention, the GPU constantly reads and writes large matrices to slow high-bandwidth memory (HBM), which becomes the main bottleneck. FlashAttention carefully organizes memory access so that most operations happen in fast on-chip memory, with minimal slow memory traffic. By reducing memory reads and writes, it removes the biggest performance bottleneck in attention.
- **Parallelization**: Better GPU utilization. FlashAttention is designed so that many GPU threads work in parallel on different tiles at the same time. This keeps the GPU fully busy instead of having parts of it idle while waiting for memory. As a result, GPU cores, memory units, and compute pipelines are all efficiently utilized, leading to large real-world speedups.

**Benefits:**
- **Memory**: 2-4x less memory usage
- **Speed**: 2-4x faster attention
- **Scalability**: Handle longer sequences
- **Quality**: Same attention results

**Implementation:**
```python
def flash_attention(q, k, v, block_size=64):
    # Process in blocks for memory efficiency
    output = []
    for i in range(0, q.size(0), block_size):
        block_q = q[i:i+block_size]
        block_k = k[i:i+block_size]
        block_v = v[i:i+block_size]
        
        # Compute attention for block
        block_output = compute_attention(block_q, block_k, block_v)
        output.append(block_output)
    
    return torch.cat(output, dim=0)
```

---

### 59. Explain the concept of model quantization.

**Answer:**

Model quantization reduces the precision of model parameters to decrease memory usage and improve inference speed.
Model quantization is the process of reducing the numerical precision of a neural network’s weights and activations (for example, from 32-bit to 8-bit or 4-bit) in order to reduce memory usage, speed up inference, and lower compute cost—while keeping accuracy as high as possible. In simple words: You store numbers with fewer bits so the model becomes smaller and faster.
Without quantization: ❌ Models don’t fit on consumer GPUs ❌ Inference is slow and expensive ❌ On-device and edge deployment becomes impossible
With quantization: ✅ Models fit on smaller hardware ✅ Inference is much faster✅ Cost drops significantly
**Types of Quantization:**
- **Post-training**: Quantize after training
- **Quantization-aware**: Train with quantization
- **Dynamic**: Quantize during inference
- **Static**: Pre-compute quantization parameters

**Precision Levels:**
- **FP32**: 32-bit floating point
- **FP16**: 16-bit floating point
- **INT8**: 8-bit integer
- **INT4**: 4-bit integer

**Benefits:**
- **Memory**: 2-4x memory reduction
- **Speed**: Faster inference
- **Deployment**: Easier deployment
- **Cost**: Lower hardware requirements

**Implementation:**
```python
# Post-training quantization
model = load_model()
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

### 60. What is the difference between dynamic and static quantization?

**Answer:**

**Dynamic Quantization:**
- **When**: During inference
- **Method**: Quantize activations on-the-fly
- **Memory**: Lower memory usage
- **Speed**: Faster inference
- **Flexibility**: Adapts to input

**Static Quantization:**
- **When**: Before inference
- **Method**: Pre-compute quantization parameters
- **Memory**: Lower memory usage
- **Speed**: Fastest inference
- **Flexibility**: Fixed quantization

**Comparison:**
| Aspect | Dynamic | Static |
|--------|---------|--------|
| **Timing** | Runtime | Pre-compute |
| **Memory** | Lower | Lower |
| **Speed** | Fast | Fastest |
| **Flexibility** | High | Low |
| **Setup** | Simple | Complex |

**When to Use:**
- **Dynamic**: Quick deployment, variable inputs
- **Static**: Production systems, fixed inputs

---

### 61. How does knowledge distillation work in language models?

**Answer:**

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model, maintaining performance while reducing size.

**Process:**
1. **Teacher Model**: Large, high-performance model
2. **Student Model**: Smaller, efficient model
3. **Distillation**: Train student to mimic teacher
4. **Transfer**: Knowledge transfer through soft targets

**Benefits:**
- **Efficiency**: Smaller, faster models
- **Performance**: Maintains teacher quality
- **Deployment**: Easier deployment
- **Cost**: Lower computational requirements

**Implementation:**
```python
class DistillationLoss:
    def __init__(self, temperature=3.0, alpha=0.7):
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, student_logits, teacher_logits, labels):
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Student predictions
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Distillation loss
        distillation_loss = F.kl_div(student_probs, soft_targets, reduction='batchmean')
        
        # Hard targets
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        return total_loss
```

---

### 62. Explain the concept of model pruning.

**Answer:**

Model pruning removes unnecessary parameters from neural networks to reduce size and improve efficiency while maintaining performance.

**Types of Pruning:**
- **Magnitude-based**: Remove smallest weights
- **Gradient-based**: Remove least important weights
- **Structured**: Remove entire neurons/layers
- **Unstructured**: Remove individual weights

**Pruning Process:**
1. **Train**: Train full model
2. **Prune**: Remove less important parameters
3. **Fine-tune**: Retrain pruned model
4. **Iterate**: Repeat if needed

**Benefits:**
- **Size**: Smaller models
- **Speed**: Faster inference
- **Memory**: Lower memory usage
- **Deployment**: Easier deployment

**Implementation:**
```python
def prune_model(model, pruning_ratio=0.5):
    # Get all weights
    weights = []
    for param in model.parameters():
        weights.extend(param.view(-1).tolist())
    
    # Find threshold
    threshold = np.percentile(weights, pruning_ratio * 100)
    
    # Prune weights
    for param in model.parameters():
        param.data[torch.abs(param.data) < threshold] = 0
    
    return model
```

---

### 63. What is the purpose of model compression techniques?

**Answer:**

Model compression reduces model size and computational requirements while maintaining performance, enabling deployment on resource-constrained devices.

**Compression Techniques:**
- **Quantization**: Reduce precision
- **Pruning**: Remove parameters
- **Distillation**: Transfer knowledge
- **Low-rank**: Factorize matrices
- **Hashing**: Share parameters

**Benefits:**
- **Deployment**: Run on edge devices
- **Cost**: Lower hardware requirements
- **Speed**: Faster inference
- **Memory**: Lower memory usage
- **Accessibility**: Wider device support

**Trade-offs:**
- **Performance**: Slight accuracy loss
- **Complexity**: More complex training
- **Time**: Additional optimization time
- **Quality**: May need fine-tuning

---

### 64. How does speculative decoding work?

**Answer:**

Speculative decoding improves inference speed by generating multiple candidate tokens in parallel and then verifying the best one.

**Process:**
1. **Draft Model**: Fast, smaller model generates candidates
2. **Target Model**: Large model verifies candidates
3. **Acceptance**: Accept verified tokens
4. **Rejection**: Reject and retry if needed

**Benefits:**
- **Speed**: 2-3x faster inference
- **Quality**: Maintains model quality
- **Efficiency**: Better GPU utilization
- **Scalability**: Works with large models

**Implementation:**
```python
def speculative_decode(draft_model, target_model, input_ids, k=4):
    # Generate draft tokens
    draft_tokens = draft_model.generate(input_ids, max_length=k)
    
    # Verify with target model
    target_logits = target_model(input_ids, draft_tokens)
    
    # Accept verified tokens
    accepted_tokens = []
    for i, token in enumerate(draft_tokens):
        if verify_token(target_logits[i], token):
            accepted_tokens.append(token)
        else:
            break
    
    return accepted_tokens
```

---

### 65. Explain the concept of model serving optimization.

**Answer:**

Model serving optimization improves inference performance, reduces latency, and increases throughput for production deployments.

**Optimization Techniques:**
- **Batching**: Process multiple requests together
- **Caching**: Cache frequent responses
- **Load Balancing**: Distribute requests
- **Auto-scaling**: Scale based on demand
- **Hardware Optimization**: Use specialized hardware

**Key Metrics:**
- **Latency**: Response time
- **Throughput**: Requests per second
- **Memory**: Memory usage
- **Cost**: Computational cost
- **Availability**: Uptime percentage

**Implementation:**
```python
class OptimizedModelServer:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.request_queue = []
        self.cache = {}
    
    def serve(self, request):
        # Check cache
        if request in self.cache:
            return self.cache[request]
        
        # Add to batch
        self.request_queue.append(request)
        
        # Process batch when full
        if len(self.request_queue) >= self.batch_size:
            return self.process_batch()
    
    def process_batch(self):
        # Process batch of requests
        batch = self.request_queue[:self.batch_size]
        results = self.model.batch_predict(batch)
        
        # Cache results
        for request, result in zip(batch, results):
            self.cache[request] = result
        
        return results
```

---

## Summary

These advanced architecture concepts are crucial for building efficient, scalable GenAI systems. Focus on understanding the trade-offs and implementation details.

**Key Takeaways:**
- Understand different model architectures
- Learn optimization techniques
- Focus on efficiency and scalability
- Practice with real implementations
- Stay updated with new developments

**Next Steps:**
- Experiment with different architectures
- Implement optimization techniques
- Build scalable systems
- Learn from production deployments
- Contribute to open-source projects

---

**Powered by [Euron](https://euron.one/) - Empowering AI Innovation Worldwide**
