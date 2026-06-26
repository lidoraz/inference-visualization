export interface GlossaryEntry {
  term: string;
  short: string;
  long: string;
}

export const glossary: Record<string, GlossaryEntry> = {
  prefill: {
    term: "Prefill",
    short: "All prompt tokens are processed in one forward pass to populate the KV cache.",
    long: "During prefill, every token in the input prompt is processed in a single forward pass through the model. This computes and stores the key/value tensors for each prompt token in the KV cache. No output token is produced yet — the result is a fully populated cache ready for the first decode step.",
  },

  decode: {
    term: "Decode",
    short: "The model generates one new token per step by reading the full KV cache.",
    long: "After prefill, the model enters the decode phase. At each step it attends over the entire KV cache (all previous tokens) and produces a probability distribution over the vocabulary, from which the next token is sampled. The new token's K/V vectors are appended to the cache, and the process repeats until an end-of-sequence token is generated or a length limit is hit.",
  },

  kvCache: {
    term: "KV Cache",
    short: "Stores key/value tensors for every past token so attention never recomputes them.",
    long: "The key-value cache holds the key and value tensors produced by each attention head for every token processed so far. By caching these tensors, subsequent decode steps can skip recomputation and simply read from the cache. The KV cache is the dominant GPU memory consumer during serving and grows linearly with sequence length and batch size.",
  },

  block: {
    term: "Block",
    short: "Block size = how many tokens' K/V fit in one block (vLLM default ~16).",
    long: "\"Block size\" is a token count: the number of consecutive tokens whose key/value tensors (across all layers and KV heads) are bundled into one fixed-size block. The block's actual memory is block_size × num_layers × 2 (K and V) × num_kv_heads × head_dim × dtype_bytes — so the per-token cost comes from the model architecture and the block just groups that many tokens. vLLM's default block size is ~16 tokens (configurable, often 8/16/32). Smaller blocks waste less space in the last partially-filled block (less internal fragmentation) but add block-table bookkeeping; larger blocks are cheaper to track but waste more per sequence. This demo uses a tiny block size of 4 so the cache visibly fills and fragments on screen — real systems use 16+.",
  },

  modelLayers: {
    term: "Model Layers",
    short: "Number of transformer layers; every layer stores its own K and V for each token.",
    long: "A transformer is a stack of identical layers (e.g. 32 for an 8B model, 80+ for the largest). Each layer has its own attention block that produces key and value tensors, so the KV cache must store K and V separately for every layer. This is why num_layers is a direct multiplier on KV-cache size: doubling the depth of the model doubles the per-token cache cost.",
  },

  kvHeads: {
    term: "KV Heads",
    short: "Number of key/value heads; grouped-query attention shares them to shrink the cache.",
    long: "Multi-head attention splits the hidden dimension into several heads. Originally each head had its own K and V (multi-head attention, MHA). Grouped-query attention (GQA) and multi-query attention (MQA) reduce the number of distinct KV heads — several query heads share one KV head — which directly shrinks the KV cache. The cache size depends on num_kv_heads, not the (often larger) number of query heads.",
  },

  headDim: {
    term: "Head Dimension",
    short: "The size of each attention head's key/value vector (hidden_dim / num_heads).",
    long: "Each attention head operates on a slice of the model's hidden dimension. head_dim is that slice size — typically 64, 128, or 256. The key and value vectors stored per token, per layer, per KV head are each head_dim elements long. So the per-token KV cost is num_layers × 2 × num_kv_heads × head_dim elements, times the bytes per element of the chosen dtype.",
  },

  bytesPerToken: {
    term: "Bytes per Token",
    short: "KV memory one token uses: layers × 2 × kv_heads × head_dim × dtype_bytes.",
    long: "Every token a request processes adds a fixed amount to the KV cache, determined entirely by the model architecture and dtype: num_layers × 2 (one each for K and V) × num_kv_heads × head_dim × dtype_bytes. A physical block holds block_size tokens, so one block costs bytes_per_token × block_size, and the whole cache costs that times the number of blocks. This is the bridge between the model's shape and the abstract 'blocks' the scheduler manages.",
  },

  blockTable: {
    term: "Block Table",
    short: "Per-request map from logical block index to physical block ID — like an OS page table.",
    long: "Each request maintains a block table that maps its logical sequence of blocks to the physical block IDs actually stored in GPU memory. This indirection, analogous to a virtual-memory page table, allows a request's KV cache to occupy non-contiguous physical blocks. When a new block is needed, the allocator assigns any free physical block and records the mapping.",
  },

  pagedAttention: {
    term: "Paged Attention",
    short: "vLLM's technique of paging the KV cache into blocks, eliminating contiguous-memory waste.",
    long: "PagedAttention is vLLM's core memory management technique. Instead of pre-allocating a contiguous buffer sized to the maximum possible sequence length, it stores KV cache entries in fixed-size blocks that can be placed anywhere in GPU memory. Attention kernels are modified to look up physical block IDs via the block table when accessing K/V data. This virtually eliminates reserved-but-unused memory and allows much higher batch sizes for a given GPU.",
  },

  internalFragmentation: {
    term: "Internal Fragmentation",
    short: "Wasted space in the last partial block; paging bounds waste to < 1 block per request.",
    long: "Internal fragmentation occurs when the last block of a sequence is only partially filled — the remaining token slots are allocated but unused. With paged attention this waste is bounded to at most one block per sequence (e.g. up to 15 tokens for a block size of 16). This contrasts with contiguous allocation, where reserving max-length memory can waste thousands of tokens worth of space per request.",
  },

  preemption: {
    term: "Preemption",
    short: "Evicting a running request's blocks when the KV cache is full and space is needed.",
    long: "When all KV cache blocks are occupied and the scheduler needs to admit a new request or grow an existing one, it must preempt a running request. Preemption frees that request's physical blocks so they can be reassigned. The preempted request is returned to the waiting queue to be rescheduled when memory becomes available.",
  },

  swapping: {
    term: "Swapping",
    short: "A preempted request's KV blocks are freed (or offloaded to CPU) and it re-queues.",
    long: "When a request is preempted, vLLM can handle it two ways: recomputation (simply free the blocks and re-queue; the request re-runs prefill when re-admitted) or swapping (copy the KV blocks to CPU RAM before freeing GPU memory, then copy them back on re-admission to avoid redoing prefill). Recomputation is simpler; swapping saves prefill cost at the expense of CPU-GPU transfer. The visualizer models the re-queue approach. In both cases the request re-enters the waiting queue until memory becomes available.",
  },

  continuousBatching: {
    term: "Continuous Batching",
    short: "Requests are admitted and retired every step, keeping the GPU busy with a changing batch.",
    long: "Traditional static batching waits for an entire batch of requests to finish before accepting new ones, leaving GPUs idle as short requests complete. Continuous batching (also called iteration-level scheduling) instead re-evaluates the batch composition at every decode step: completed requests are immediately retired and new waiting requests are admitted. This keeps GPU utilization high and dramatically improves throughput.",
  },

  tokenBudget: {
    term: "Token Budget",
    short: "Per-step cap on the total number of tokens the scheduler will process in one batch step.",
    long: "The token budget (vLLM's max_num_batched_tokens) is a configurable limit on how many tokens are included in a single forward pass. Prefilling a long prompt consumes many tokens at once, so the budget prevents a single bulky request from starving others. The scheduler sums token counts across all selected requests and stops adding candidates once the budget is reached — lower it and long prompts get deferred to later steps so they can't stall the existing batch's decode work.",
  },

  maxBatchSize: {
    term: "Max Batch Size",
    short: "The maximum number of requests allowed to run concurrently in the active batch.",
    long: "Max batch size caps how many requests the scheduler can include in the running batch at any one time (vLLM calls this max_num_seqs). It bounds memory usage for metadata and attention masks and prevents thrashing when too many sequences compete for KV cache blocks. The scheduler will not admit new requests beyond this limit even if the token budget and block pool have remaining capacity. Real deployments run this at 256 or more to keep the GPU saturated; a value of 1 serializes requests one at a time and wastes most of the hardware — useful only for demonstrating the lifecycle of a single request.",
  },

  vocabSampling: {
    term: "Vocab Sampling",
    short: "Each decode step samples the next token from a full vocab distribution (~32k–200k).",
    long: "At the end of each decode step the model's language-model head projects the hidden state into a logit vector of size |V| (e.g. ~32k for GPT-2-era models, ~128k for Llama 3, up to ~200k for GPT-4o). A softmax converts these to probabilities, and a sampling strategy (greedy, top-k, top-p, temperature) selects the next token id. This id is then decoded back to text and appended to the output.",
  },

  temperature: {
    term: "Temperature",
    short: "Scales logits before softmax: <1 sharpens the distribution, >1 flattens it.",
    long: "Temperature divides the logits before the softmax. Below 1.0 it sharpens the distribution — probability mass concentrates on the top tokens, making output more deterministic and focused. Above 1.0 it flattens the distribution, spreading mass to less-likely tokens for more diverse/creative (and riskier) output. At 0 it collapses to greedy (always the argmax). It's the simplest knob for the determinism-vs-creativity tradeoff.",
  },

  topK: {
    term: "Top-k",
    short: "Keeps only the k most-likely tokens, then renormalizes and samples among them.",
    long: "Top-k sampling truncates the distribution to the k highest-probability tokens, zeroes the rest, renormalizes, and samples from that shortlist. It caps how far down the tail the model can reach (k=1 is greedy; large k approaches full sampling). It bounds worst-case nonsense but uses a fixed cutoff regardless of how peaked or flat the distribution is — which is what top-p addresses.",
  },

  topP: {
    term: "Top-p (nucleus)",
    short: "Keeps the smallest set of top tokens whose probabilities sum to p, then samples.",
    long: "Top-p (nucleus) sampling keeps the smallest set of most-likely tokens whose cumulative probability reaches p (e.g. 0.9), then renormalizes and samples from that set. Unlike top-k's fixed count, the nucleus size adapts to the distribution: a confident, peaked step keeps few tokens; an uncertain, flat step keeps many. It's the most common modern default, often combined with a temperature.",
  },

  greedy: {
    term: "Greedy Decoding",
    short: "Always pick the single most-likely token (argmax) — fully deterministic, no sampling.",
    long: "Greedy decoding skips sampling entirely and always takes the argmax — the highest-probability token at every step. It's fully deterministic (same prompt → same output) and equivalent to temperature 0 or top-k 1. It maximizes local likelihood but can be repetitive or get stuck, and it explores none of the distribution — which is why interactive generation usually samples with temperature/top-p instead.",
  },

  tokenizer: {
    term: "Tokenizer",
    short: "Splits text into subword tokens; a naive splitter stands in for a real BPE tokenizer.",
    long: "A tokenizer converts raw text into a sequence of integer token ids the model can process. Production models use algorithms like Byte-Pair Encoding (BPE) to build a vocabulary of common subword units. In this visualizer a simplified word/subword splitter stands in for a real tokenizer, chunking words into pieces of at most MAX_CHARS characters and assigning sequential integer ids.",
  },

  waitingQueue: {
    term: "Waiting Queue",
    short: "Requests that have arrived but not yet been admitted to the running batch.",
    long: "Newly submitted requests enter the waiting queue, where they sit until the scheduler decides to admit them. A request waits when the running batch is at capacity (max batch size reached), the token budget would be exceeded, or there are not enough free KV cache blocks to cover its prefill. The scheduler processes the waiting queue every step, admitting as many requests as constraints allow.",
  },

  runningQueue: {
    term: "Running Queue",
    short: "Requests currently being prefilled or decoded in the active batch.",
    long: "The running queue holds all requests currently participating in the forward pass — both those still in the prefill phase and those actively decoding. Each step, the scheduler extends every running request by one decode token (or completes their prefill), updates the KV cache, and checks for completions. Requests leave the running queue either by finishing or by being preempted.",
  },

  // ── Stage 4: PD Disaggregation ───────────────────────────────────────────

  pdDisaggregation: {
    term: "PD Disaggregation",
    short: "Splitting prefill and decode onto separate workers for independent scaling.",
    long: "Prefill-decode disaggregation (PD disaggregation) runs prefill steps and decode steps on separate GPU workers. Prefill is compute-bound and bursty (a whole prompt in one shot), while decode is memory-bandwidth-bound and steady (one token per step). By separating them, each worker can be sized and optimized independently — prefill workers can handle larger prompts without stalling decode workers, improving overall throughput and latency.",
  },

  prefillWorker: {
    term: "Prefill Worker",
    short: "A GPU specialized in processing prompt tokens in a single forward pass.",
    long: "In a disaggregated setup, the prefill worker handles the initial forward pass over all prompt tokens. It is optimized for high compute throughput and can handle large batches of tokens simultaneously. Once prefill completes, the resulting KV cache tensors are transferred to a decode worker, and the prefill worker is freed for the next incoming request.",
  },

  decodeWorker: {
    term: "Decode Worker",
    short: "A GPU specialized in autoregressive token generation, one token per step.",
    long: "The decode worker receives the KV cache from the prefill worker and takes over autoregressive generation. It runs one decode step per tick, attending over the accumulated KV cache to produce the next token. Decode workers are optimized for memory-bandwidth efficiency rather than raw compute, since each step only processes a single new token against a large cache.",
  },

  kvTransfer: {
    term: "KV Transfer",
    short: "Shipping KV cache tensors from the prefill worker to the decode worker.",
    long: "After a prefill worker completes the initial forward pass, the computed key/value tensors for all prompt tokens must be sent over the network (or NVLink) to the decode worker that will continue generation. This KV transfer is a key cost in disaggregated serving: the bandwidth and latency of the transfer must be smaller than the time saved by specializing the two worker types.",
  },

  weightReplication: {
    term: "Weight Replication",
    short: "Each pool holds its own full weight copy — replication across roles, not 2× hardware.",
    long: "Because the prefill and decode pools are separate GPUs, you can't share weights across them — each pool must hold its own full copy of the model in HBM. But this is a replication of parameters across role-partitioned hardware, NOT a doubling of your GPU count, memory, or cost. The same 8 GPUs that run one tensor-parallel-8 instance (one copy sharded over 8, each GPU holding weights/8) can instead be split into prefill-TP4 + decode-TP4: now two full copies exist, but each is sharded over only 4 GPUs (weights/4 each) — same 8 GPUs, same spend. The pools need not be symmetric; you tune the prefill:decode ratio to the workload. The real feasibility limit is per-pool: each pool's GPUs must collectively fit a full copy. A model that already nearly fills your whole budget as a single instance can't be partitioned into pools on the same hardware and must scale out. Whether replicated weights are even a meaningful overhead depends on how weight memory compares to KV-cache memory — at long context the KV cache often dominates.",
  },

  // ── Stage 5: Parallelism ─────────────────────────────────────────────────

  tensorParallel: {
    term: "Tensor Parallelism",
    short: "Each GPU holds a shard of every layer's weights; all-reduce syncs after each layer.",
    long: "Tensor parallelism (TP) splits individual weight matrices across GPUs. For a transformer layer, the attention heads or FFN weight columns are divided evenly — GPU 0 holds heads 0–N/2, GPU 1 holds heads N/2–N, etc. Each GPU computes its shard of the forward pass, then an all-reduce collective sums the partial results before moving to the next layer. TP reduces per-GPU memory and enables larger models, at the cost of all-reduce communication every layer.",
  },

  pipelineParallel: {
    term: "Pipeline Parallelism",
    short: "Each GPU holds a contiguous subset of layers; requests flow through GPUs in sequence.",
    long: "Pipeline parallelism (PP) assigns different model layers to different GPUs: GPU 0 runs layers 0–7, GPU 1 runs layers 8–15, and so on. A request's activations flow from GPU to GPU as a pipeline. To keep all GPUs busy, the batch is split into micro-batches so GPU 1 works on micro-batch 1 while GPU 0 starts micro-batch 2. PP reduces per-GPU memory proportional to pipeline depth, but introduces pipeline-fill latency (bubble overhead) at the start and end of each batch.",
  },

  expertParallel: {
    term: "Expert Parallelism",
    short: "MoE experts are distributed across GPUs; tokens route to the GPU holding their expert.",
    long: "Expert parallelism (EP) is a parallelism strategy for Mixture-of-Experts (MoE) models. Each GPU hosts a subset of the expert FFN blocks. During inference, each token's router selects a top-k set of experts, and the token's activations are dispatched (all-to-all communication) to the GPUs that hold those experts. EP lets the total number of experts grow with GPU count without increasing per-GPU memory, but requires efficient routing and collective communication.",
  },

  dataParallel: {
    term: "Data Parallelism",
    short: "Each GPU holds a full model copy and processes a different subset of requests.",
    long: "Data parallelism (DP) replicates the full model on every GPU, then splits the request batch across replicas. GPU 0 handles requests 0, 4, 8…; GPU 1 handles requests 1, 5, 9…; and so on. There is no inter-GPU communication during the forward pass — each replica runs independently. DP scales throughput linearly with GPU count and is the simplest strategy, but requires enough memory on each GPU to hold the full model.",
  },

  gpu: {
    term: "GPU",
    short: "A massively parallel processor used to run LLM forward passes and store KV caches.",
    long: "Graphics Processing Units (GPUs) are the dominant hardware for LLM inference. Their thousands of CUDA cores execute matrix multiplications in parallel, making them well-suited for transformer attention and FFN operations. GPU memory (HBM) stores model weights, KV caches, and activations. Memory capacity and memory bandwidth are the two key constraints in LLM serving — capacity limits how many parameters and KV tokens fit, while bandwidth limits how fast weights can be read for each decode step.",
  },

  // ── Stage 6: Model Features ──────────────────────────────────────────────

  fp8: {
    term: "FP8 Quantization",
    short: "8-bit floating point: half the memory of FP16, enabling ~2x throughput at lower precision.",
    long: "FP8 is an 8-bit floating-point format (1 byte per element) compared to FP16's 2 bytes. Storing model weights and KV cache entries in FP8 roughly halves their memory footprint, which lets twice as many weights or KV tokens fit in the same GPU memory. Modern GPUs with native FP8 matrix-multiply units (NVIDIA Hopper/H100, Ada/L40S, Blackwell/B200, and AMD MI300X) benefit in throughput as well. The trade-off is reduced numerical precision, which requires careful calibration or quantization-aware training to avoid accuracy loss.",
  },

  quantization: {
    term: "Quantization",
    short: "Lowering weight/activation precision (FP16→FP8→INT4) to cut memory and speed up inference.",
    long: "Quantization converts a model's weights or activations from high-precision formats (FP32, FP16) to lower-precision formats (FP8, INT8, INT4). Lower precision means smaller storage per parameter, less memory bandwidth per operation, and often faster matrix multiplications on supporting hardware. The challenge is that reducing precision introduces rounding error, which can degrade model quality. Techniques like GPTQ, AWQ, and SmoothQuant aim to minimize this degradation through calibration on representative data.",
  },

  moe: {
    term: "Mixture of Experts (MoE)",
    short: "A model architecture with many expert FFN blocks; only top-k are activated per token.",
    long: "Mixture of Experts (MoE) replaces a standard dense FFN layer with a collection of E expert networks plus a router. The router examines each token's hidden state and selects the top-k experts (typically 2 out of 8, 16, or more) to process that token. Only the selected experts run — the rest are skipped. This means total model parameters can be very large (more experts = more capacity/knowledge) while per-token compute (active parameters) remains proportional to top-k, not total experts. MoE enables models like Mixtral and DeepSeek to have billions of total parameters with much lower inference cost per token.",
  },

  expertRouting: {
    term: "Expert Routing",
    short: "The router assigns each token to its top-k experts based on learned gating weights.",
    long: "Expert routing is the mechanism by which an MoE layer decides which experts process each token. A small learned router (typically a linear layer) computes a score for each expert given the token's hidden state. The top-k highest-scoring experts are selected, and the token's activations are sent to those experts (in parallel or via dispatch in expert-parallel setups). The router's gating weights are trained end-to-end, learning to specialize experts for different types of tokens or semantic domains.",
  },

  activeParams: {
    term: "Active Parameters",
    short: "The fraction of total model parameters actually used when processing a single token.",
    long: "In a dense model, every parameter participates in every forward pass — active parameters equal total parameters. In an MoE model, only the router and the top-k selected experts are active for each token. For example, a model with 16 experts of 1B params each has 16B total parameters, but if top-2 are selected, only ~2B params are active per token. This ratio (active/total) determines per-token compute cost (FLOPs and memory bandwidth), while total parameters determine model capacity and storage cost.",
  },

  prefixCache: {
    term: "Prefix Cache",
    short: "Reuses KV for a shared prompt prefix so it isn't recomputed for every request.",
    long: "When many requests share an identical leading prefix — a system prompt, few-shot examples, or earlier turns of a conversation — the key/value tensors for that prefix are identical. A prefix cache stores those KV entries once and lets later requests reuse them instead of re-running prefill over the shared tokens. SGLang has RadixAttention on by default; vLLM exposes it via enable_prefix_caching (enabled with the modern V1 engine). Both engines differ in data structure: vLLM hashes blocks of tokens and matches them in a flat block pool (with LRU eviction), while SGLang structures the entire cache as a radix tree where prefix sharing is the native shape. Functionally both reuse shared-prefix KV — the radix-tree-vs-hash distinction is an implementation detail, not a capability gap.",
  },

  radixAttention: {
    term: "RadixAttention",
    short: "SGLang's radix-tree KV cache: shared prefixes share nodes, branching where they differ.",
    long: "RadixAttention is SGLang's approach to KV-cache reuse. Instead of a flat pool of blocks with a separate hash table for prefix matching, SGLang stores the cache as a radix tree (a compressed trie) keyed by token sequences. Requests that share a prefix walk the same path from the root and only branch at the first differing token, so shared KV is stored exactly once and reused automatically. Eviction is LRU on the tree's leaves. This makes prefix sharing the native behavior rather than an add-on, which is especially powerful for multi-turn chat, few-shot prompting, and tree-of-thought style branching.",
  },

  lruEviction: {
    term: "LRU Eviction",
    short: "When the cache is full, the least-recently-used tree leaf is evicted first.",
    long: "SGLang's radix tree can grow beyond available memory, so it evicts entries to make room. It uses a least-recently-used policy on the tree's leaf nodes: the branch whose tokens were accessed longest ago is freed first. Interior nodes shared by active requests are protected because they remain referenced. This keeps hot shared prefixes (like a common system prompt) resident while cold, one-off continuations are reclaimed.",
  },

  dpAttention: {
    term: "DP Attention",
    short: "Each rank runs attention on its own requests, holding distinct KV (no replication).",
    long: "Data-parallel attention assigns whole requests to ranks: each rank holds the complete KV for a different subset of requests and runs the attention layer locally, with no cross-rank KV duplication. This specifically fixes the MLA case — DeepSeek-style multi-head latent attention compresses KV into one small per-token latent vector that can't be cleanly sharded across heads, so plain tensor parallelism ends up replicating it on every rank. DP attention was pioneered in SGLang for DeepSeek serving and is now also supported by vLLM (each DP engine keeps an independent KV cache). The MoE FFN is handled separately by expert parallelism.",
  },

  programmableFrontend: {
    term: "Programmable Frontend",
    short: "Script multi-step LLM programs; the runtime shares KV across branches automatically.",
    long: "SGLang exposes a Python DSL (the 'SGLang' in the name — Structured Generation Language) for writing LLM programs: prompts with control flow, parallel sub-calls, constrained decoding, and reusable prefixes. Because the frontend understands the program structure, the runtime can automatically share KV across branches (via the radix tree), batch parallel calls, and overlap work. This is a different model from sending independent opaque requests — the engine sees the shared structure and exploits it.",
  },

  overlappedScheduling: {
    term: "Overlapped Scheduling",
    short: "Overlaps CPU scheduling and GPU compute (and comm) so the GPU rarely stalls between steps.",
    long: "In a naive loop the GPU sits idle while the CPU prepares the next batch (tokenization, scheduling, building block tables). SGLang's cache-aware, overlapped scheduler runs that CPU work for step N+1 while the GPU is still computing step N, and overlaps expert-parallel all-to-all communication with compute. This keeps the GPU continuously busy, improving throughput — conceptually similar in spirit to the continuous batching idea, but applied to the scheduler/compute/communication pipeline itself.",
  },

  ttft: {
    term: "TTFT",
    short: "Time To First Token — how long a request waits before it sees its first output token.",
    long: "Time To First Token is the latency from when a request arrives to when it emits its first decoded token. It covers queue wait + prefill of the whole prompt, so long prompts and a busy scheduler both inflate it. TTFT is the headline interactivity metric — it's what a user feels as 'how long until the model starts responding.' In this demo it's measured as firstTokenTick − arrivalTick.",
  },

  itl: {
    term: "ITL",
    short: "Inter-Token Latency — the time between consecutive output tokens during decode.",
    long: "Inter-Token Latency (also called TPOT, time per output token) is the steady-state gap between decoded tokens once generation is underway. It sets the streaming speed a user sees ('words per second'). ITL is memory-bandwidth-bound and grows as the batch gets larger or the KV cache gets longer. Tight TTFT and tight ITL pull in different directions, which is exactly the tension PD disaggregation and chunked prefill address.",
  },

  throughput: {
    term: "Throughput",
    short: "Total output tokens generated per unit time across all requests in the batch.",
    long: "Throughput is the aggregate token-generation rate of the whole server (tokens/second across every active request), as opposed to the per-request latency that TTFT and ITL measure. Continuous batching, larger batch sizes, and higher GPU utilization raise throughput — often at some cost to per-request latency. The real optimization target is usually goodput: throughput that actually meets the latency SLOs, not raw tokens/second.",
  },

  speculativeDecoding: {
    term: "Speculative Decoding",
    short: "A small draft model guesses k tokens; the target verifies them in one pass.",
    long: "Speculative decoding speeds up decode without changing the output. A small, fast draft model proposes the next k tokens; the large target model then verifies all k in a single forward pass (the same cost as generating one token). The target accepts the longest correct prefix of the guess and corrects the first wrong token, so every target pass yields 1 + (accepted) tokens instead of 1. Because verification is parallel and decode is memory-bandwidth-bound (the target's weights get read once for k positions), this is a free speedup when the draft is accurate — the output distribution is provably identical to plain sampling.",
  },

  draftModel: {
    term: "Draft Model",
    short: "A small fast model that proposes candidate tokens for the target to verify.",
    long: "The draft model is a small, cheap model (a tiny LLM, a few extra heads as in Medusa, or the target attending to a draft tree as in EAGLE) that quickly guesses the next several tokens. Its guesses don't need to be perfect — wrong ones are caught by verification — but the closer its distribution is to the target's, the higher the acceptance rate and the bigger the speedup. A draft that's too large eats the savings; too inaccurate and most guesses are rejected.",
  },

  verification: {
    term: "Verification",
    short: "The target model checks all k draft tokens in one parallel forward pass.",
    long: "Verification runs the large target model over the prompt plus all k drafted tokens in a single forward pass — the same latency as decoding one token, because the target's weights are read once and the k positions are processed in parallel. The target compares its own distribution at each position against the draft and accepts the longest prefix that matches (under the sampling rule), then resamples the first mismatch. This is what guarantees the final output is identical to having decoded normally.",
  },

  acceptanceRate: {
    term: "Acceptance Rate",
    short: "Fraction of drafted tokens the target accepts — sets the realized speedup.",
    long: "The acceptance rate is how often the target agrees with the draft's guesses. It's the key lever on speculative decoding's payoff: with draft length k and acceptance rate a, each target pass yields more than one token (a rough upper bound is 1 + a·k; the exact expectation is lower at moderate a). A high acceptance rate (a good, well-aligned draft on predictable text) can give 2–3× decode speedups; a low rate wastes draft compute and the verification falls back toward one-token-at-a-time. It depends on the draft's quality and how predictable the text is.",
  },

  chunkedPrefill: {
    term: "Chunked Prefill",
    short: "Split a long prompt's prefill into token-budget-sized chunks, interleaved with decodes.",
    long: "Normally a long prompt is prefilled in one big forward pass, which monopolizes a step and stalls every other request's decode (a TTFT spike for them). Chunked prefill (Sarathi-Serve) instead slices the prefill into chunks that fit the per-step token budget and interleaves those chunks with ongoing decode tokens in the same batch. This bounds the inter-token-latency spikes long prompts used to cause and keeps decode flowing — and it's a big reason colocated serving stays competitive with PD disaggregation. In this demo prefill is one tick for simplicity; chunked prefill is the production refinement of that step.",
  },

  slidingWindow: {
    term: "Sliding-Window Attention",
    short: "Each token attends only to the last W tokens, so KV cache stays bounded.",
    long: "In sliding-window attention each token attends only to a fixed-size window of the most recent W tokens (e.g. 4096) rather than the entire history. Because the model never looks further back than W, the KV cache for a sequence can be capped at W tokens — older entries are evicted as the window slides forward. This turns KV memory from O(sequence length) into O(W), which is what makes very long contexts (100k–1M tokens) affordable to serve. The tradeoff is that information older than W tokens can only propagate forward indirectly through the layers; some architectures (e.g. Gemma 2) interleave sliding-window and full-attention layers to balance this.",
  },

  attentionVariant: {
    term: "Attention Variants (MHA/GQA/MLA)",
    short: "How K/V heads are shared (or compressed) to shrink the KV cache.",
    long: "The KV cache size is set by how many key/value heads each layer stores. Multi-Head Attention (MHA) gives every query head its own KV head — largest cache. Grouped-Query Attention (GQA) shares one KV head across a group of query heads (e.g. 8 query heads → 1 KV head), cutting the cache several-fold with little quality loss — now the common default (Llama 3, etc.). Multi-head Latent Attention (MLA, from DeepSeek) goes further, compressing K/V into a single small latent vector per token that's decompressed on the fly — the smallest cache, but it doesn't shard cleanly under tensor parallelism (which is why it pairs with DP-attention). Fewer/compressed KV heads = smaller cache = longer contexts and bigger batches.",
  },

  mha: {
    term: "Multi-Head Attention (MHA)",
    short: "Every query head has its own K/V head — the largest KV cache.",
    long: "Multi-Head Attention is the original transformer attention: the hidden dimension is split into H heads, and each head has its own query, key, and value projections. The KV cache must store K and V for all H heads per token, making it the most memory-hungry option. GQA and MLA were introduced specifically to shrink this.",
  },

  gqa: {
    term: "Grouped-Query Attention (GQA)",
    short: "Several query heads share one K/V head, shrinking the cache several-fold.",
    long: "Grouped-Query Attention reduces the number of distinct KV heads: query heads are split into groups, and each group shares a single key/value head (e.g. 32 query heads sharing 8 KV heads). This cuts the KV cache by the grouping factor with negligible quality loss, which is why it's the common default in modern models (Llama 3, Qwen, etc.). Multi-Query Attention (MQA) is the extreme case where all query heads share one KV head.",
  },

  mla: {
    term: "Multi-head Latent Attention (MLA)",
    short: "Compresses K/V into one small latent vector per token — the smallest cache.",
    long: "Multi-head Latent Attention, introduced by DeepSeek, compresses the key/value tensors into a single low-rank latent vector per token, stored once and decompressed on the fly during attention. This gives the smallest KV cache of the three variants. The catch: the latent vector can't be cleanly sharded across heads, so under tensor parallelism it ends up replicated on every rank — which is exactly why MLA models are served with data-parallel attention (DP-attention) instead. See the SGLang stage.",
  },

  hiddenState: {
    term: "Hidden State (h)",
    short: "The residual-stream vector output by the last transformer layer — the model's full internal representation of the sequence so far.",
    long: "After the final transformer layer, the residual stream holds a vector h of size d_model (e.g. 7168 for DeepSeek-V3). This single vector encodes everything the model knows about the context. Both the LM head and the MTP module read from this same h — they are siblings, not a chain. The LM head projects h up to vocab size to produce token logits. The MTP module combines h with an embedding of the last predicted token to draft the next position.",
  },

  embeddingLookup: {
    term: "Token Embedding — emb(t)",
    short: "A d_model-wide vector looked up from the embedding table for a single token id.",
    long: "After the LM head picks the next token (argmax or sample), the logits are discarded and the winning token id is looked up in the embedding table — a matrix of shape vocab_size × d_model. The result is one row of that matrix: a dense vector of size d_model. This is what gets fed into the MTP module alongside h. The logits never travel forward; only the single chosen id matters, and it re-expands to d_model through this lookup. This is also why the embedding table can be shared between the backbone's input layer and the MTP module: both are doing the same token-id → d_model operation.",
  },

  lmHeadTranspose: {
    term: "LM Head — embᵀ(h)",
    short: "Projects h to vocab-size logits using the embedding matrix transposed.",
    long: "The LM head is a linear projection from d_model to vocab_size that turns the hidden state h into a logit for every token. In most modern LLMs (including DeepSeek-V3) this projection reuses the embedding matrix transposed — so emb maps token ids to d_model vectors, and emb⁻¹ (the LM head) maps d_model vectors back to vocab scores. This weight-tying halves the parameter count at the vocabulary boundary and is why the MTP module can 'share the LM head with the backbone': there is only one matrix, used in both directions.",
  },

  mtp: {
    term: "MTP Module",
    short: "One extra transformer block that drafts the next token using h + emb(last token).",
    long: "The MTP (Multi-Token Prediction) module is a single transformer block — separate from the backbone's N layers — that takes two inputs: the backbone's final hidden state h, and the embedding of the token the backbone just predicted. These are concatenated, projected down to d_model by a learned matrix, and fed through the transformer block to produce a new hidden state h′. That h′ is then passed through the shared LM head to get draft logits. For k=3, this one module is called recursively 3 times, each time feeding the previous h′ and the embedding of the previous draft. The embedding table and LM head are shared with the backbone; only the transformer block and the projection matrix are unique to the MTP module.",
  },
};
