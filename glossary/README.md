# ML Glossary

**Every machine learning term you'll run into, in plain words.** 300+ terms, from *epoch* to *PEFT*, grouped by topic. Wherever a term appears in this repo's code, there's a 🔨 link so you can watch it work.

> **Tip:** press **Ctrl+F** (or **Cmd+F** on a Mac) to search for a term. For a map of the training *methods* themselves, see [**AI Training Methods**](<../AI Training Methods/TRAINING_METHODS.md>).

## Contents

1. [The basics](#1-the-basics)
2. [Ways to learn (paradigms)](#2-ways-to-learn-paradigms)
3. [Neural network building blocks](#3-neural-network-building-blocks)
4. [How training works](#4-how-training-works)
5. [Data](#5-data)
6. [Transformers and language models](#6-transformers-and-language-models)
7. [How LLMs are trained](#7-how-llms-are-trained)
8. [Fine-tuning and PEFT](#8-fine-tuning-and-peft)
9. [Generative models](#9-generative-models)
10. [Reinforcement learning (RL)](#10-reinforcement-learning-rl)
11. [Evaluation](#11-evaluation)
12. [Hardware, scale and deployment](#12-hardware-scale-and-deployment)
13. [Safety and responsible AI](#13-safety-and-responsible-ai)

---

## 1. The basics

- **Artificial intelligence (AI)**: computers doing tasks that normally take human intelligence, such as recognizing images, understanding language or playing games.
- **Machine learning (ML)**: the part of AI where a computer *learns patterns from data* instead of following rules a person wrote by hand.
- **Deep learning**: machine learning with *neural networks* that have many layers. Almost every modern AI (ChatGPT, image generators, self-driving) is deep learning.
- **Model**: the thing that learns. It takes an input (an image, some text) and produces an output (a label, the next word). It's a big math function with adjustable numbers inside.
- **Parameters / weights**: the adjustable numbers inside a model. Training means finding good values for them. 🔨 The digit recognizer has 421,642 of them, saved in [`digit_model.pt`](../digit_model.pt).
- **Bias (parameter)**: an extra number added in each neuron that shifts its output, like the *b* in *y = mx + b*. (Not the same as [dataset bias](#13-safety-and-responsible-ai).)
- **Training**: showing a model data and adjusting its weights so its outputs get better. 🔨 [`train.py`](../train.py)
- **Inference**: using a trained model to make predictions. No learning happens. 🔨 [`predict.py`](../predict.py), and the [live demo](https://kiddo4993.github.io/train-ai-from-scratch/), which runs inference in your browser.
- **Input / features**: what you give the model. For an image, each pixel is a feature. 🔨 The Snake agent's features are 11 yes/no facts, like "danger ahead" ([`snake_rl/snake.py`](../snake_rl/snake.py)).
- **Label / target**: the correct answer for an example, like "this image is a 7".
- **Prediction / output**: the model's answer.
- **Dataset**: the collection of examples used to train and test a model. 🔨 MNIST (digits), Tiny Shakespeare (text), and [`my_digits/`](../my_digits) (your own handwriting).
- **Example / sample**: one item in a dataset, such as one image with its label.
- **Loss (cost)**: one number that says *how wrong* the model is. Training tries to make it smaller.
- **Loss function**: the formula that calculates the loss. Different tasks use different ones (see [cross-entropy](#4-how-training-works) and [MSE](#4-how-training-works)).
- **Epoch**: one full pass through the whole training set. 🔨 `EPOCHS = 10` in [`train.py`](../train.py).
- **Batch / mini-batch**: a small group of examples processed together before one weight update. 🔨 `BATCH_SIZE = 64` in [`train.py`](../train.py).
- **Batch size**: how many examples are in a batch. Bigger batches are faster on a GPU but use more memory.
- **Step / iteration**: one weight update (one batch).
- **Hyperparameter**: a setting *you* choose before training, like learning rate, batch size or number of layers. The model doesn't learn it. 🔨 The CAPITALIZED settings at the top of every training script.
- **Overfitting**: the model memorizes its training data and does badly on new data. Like memorizing the answer key instead of learning the subject.
- **Underfitting**: the model is too simple, or trained too little, to learn the pattern at all.
- **Generalization**: how well a model does on data it has *never seen*. This is the real goal.
- **Training / validation / test set**: three separate piles of data. The model learns from the **training** set, you tune settings using the **validation** set, and you report the final score on the **test** set, which you touch only at the end. 🔨 [`finetune.py`](../finetune.py) holds back 25% of your drawings as a test set.
- **Baseline**: a simple reference to compare against. 🔨 Random moves (0.1 apples) are the baseline for the Snake agents.
- **Algorithm**: a step-by-step procedure. In ML, the training method (like gradient descent or Q-learning).
- **Classification**: predicting a *category* (which digit, spam or not spam). 🔨 The digit recognizer.
- **Regression**: predicting a *number* (a house price, tomorrow's temperature).
- **Clustering**: grouping similar examples together without labels.
- **Classical ML**: older, non-neural methods, still widely used on tables of data: linear regression, logistic regression, decision trees, random forests, support vector machines (SVMs), k-nearest neighbors (k-NN), XGBoost.

## 2. Ways to learn (paradigms)

- **Supervised learning**: learning from examples *with* correct answers (labels). 🔨 [Digit recognizer](../train.py).
- **Unsupervised learning**: finding structure in data *without* labels, such as groups or compressed representations. 🔨 [VAE](../autoencoder/).
- **Self-supervised learning**: the data labels itself. For example, hide the next word and predict it. This is how LLMs learn from the whole internet without anyone labeling it. 🔨 [Tiny GPT](../language_model/).
- **Semi-supervised learning**: a few labeled examples plus lots of unlabeled ones.
- **Reinforcement learning (RL)**: learning by trial and error from *rewards*, with no correct answers given. 🔨 [Snake AI](../snake_rl/). See [section 10](#10-reinforcement-learning-rl).
- **Transfer learning**: reusing a model trained on one task as the starting point for another. 🔨 [`finetune.py`](../finetune.py) starts from the MNIST model.
- **Fine-tuning**: continuing to train a pretrained model on new, usually smaller, data. See [section 8](#8-fine-tuning-and-peft).
- **Imitation learning / behavior cloning**: learning by copying an expert's actions.
- **Few-shot learning**: learning a new task from just a handful of examples. **One-shot** means one example; **zero-shot** means none (just a description).
- **In-context learning**: an LLM picking up a task from examples in the *prompt*, with no weight updates.
- **Meta-learning**: "learning to learn", where a model trains to pick up new tasks quickly.
- **Contrastive learning**: learning by pulling similar things together and pushing different things apart (two crops of the same photo should match). CLIP and SimCLR work this way.
- **Continual / online learning**: learning from a stream of new data over time.
- **Catastrophic forgetting**: when learning something new makes a network forget what it knew. 🔨 [`finetune.py`](../finetune.py) checks MNIST accuracy after fine-tuning to measure it.
- **Federated learning**: training across many devices (like phones) without collecting their raw data in one place.
- **Active learning**: the model picks which examples a human should label next, to save labeling effort.
- **Curriculum learning**: training on easy examples first, then harder ones, like a school curriculum.
- **Multi-task learning**: one model trained on several tasks at once, sharing what it learns.

## 3. Neural network building blocks

- **Neural network**: a model made of layers of simple units ("neurons"), loosely inspired by the brain. Each neuron multiplies its inputs by weights, adds them up, and applies an activation function.
- **Neuron / unit**: one small calculation: a weighted sum plus a bias, then an activation.
- **Layer**: a group of neurons that work in parallel. Data flows through layers one after another.
- **Input / hidden / output layer**: the first layer receives the data, the middle ("hidden") layers do the work, and the last layer gives the answer.
- **Architecture**: the design of a network: which layers, how many, and how they connect.
- **Fully connected / dense / linear layer**: every input connects to every output. 🔨 `nn.Linear(3136, 128)` in [`train.py`](../train.py).
- **Multilayer perceptron (MLP)**: a network made only of fully connected layers. 🔨 The DQN's `QNet` (14 → 128 → 128 → 3) in [`snake_dqn/train_dqn.py`](../snake_dqn/train_dqn.py).
- **Activation function**: the non-linear step after each layer. Without it, a deep network could only learn straight-line patterns.
  - **ReLU**: `max(0, x)`. Simple, fast and the most common. 🔨 Used in the digit recognizer.
  - **GELU / SiLU (Swish)**: smoother versions of ReLU, common in transformers.
  - **Sigmoid**: squashes any number into 0–1.
  - **Tanh**: squashes any number into −1 to 1.
  - **Softmax**: turns a list of scores into probabilities that add up to 1. 🔨 The 10 confidence bars in the [live demo](https://kiddo4993.github.io/train-ai-from-scratch/).
- **Logits**: the raw scores a model outputs *before* softmax.
- **Convolutional neural network (CNN)**: a network for images that slides small filters across the picture to detect edges, then shapes, then whole objects. 🔨 The digit recognizer is a CNN.
- **Convolution / kernel / filter**: a small grid of weights (like 3×3) slid over an image, producing a "feature map" that lights up where its pattern appears.
- **Channel**: one layer of an image or feature map. A color photo has 3 (red, green, blue); MNIST has 1. 🔨 The CNN goes from 1 channel to 32, then 64.
- **Stride / padding**: how far a filter jumps each step, and the border added so edges are covered.
- **Pooling (max pooling)**: shrinking a feature map by keeping only the largest value in each small window. 🔨 `MaxPool2d` in [`train.py`](../train.py).
- **Flatten**: unrolling a 2-D grid into one long list so a dense layer can use it.
- **Recurrent neural network (RNN)**: a network that reads a sequence one step at a time, keeping a memory. Mostly replaced by transformers.
- **LSTM / GRU**: RNN types with "gates" that help them remember over longer sequences.
- **Transformer**: the architecture behind modern LLMs, built on *attention*. See [section 6](#6-transformers-and-language-models).
- **Embedding**: a learned list of numbers (a vector) that represents a token, word, image or anything else. Similar things get similar vectors. 🔨 `nn.Embedding` in [`language_model/train_lm.py`](../language_model/train_lm.py).
- **Vector / tensor**: a vector is a list of numbers; a tensor is the general version with any number of dimensions (a matrix is a 2-D tensor). PyTorch does everything with tensors.
- **Dimension / shape**: the size of a tensor, like `(64, 1, 28, 28)` for a batch of 64 grayscale 28×28 images.
- **Residual / skip connection**: adding a layer's input straight to its output, which lets very deep networks train. Used in ResNet and every transformer.
- **Normalization layer**: rescales values inside the network to keep training stable.
  - **Batch norm**: normalizes across the batch. Common in CNNs.
  - **Layer norm**: normalizes across each example's features. 🔨 Used in the tiny GPT.
  - **RMSNorm**: a simpler, faster layer norm used in many modern LLMs.
- **Dropout**: randomly switching off some neurons during training so the network can't rely on any single one. A regularizer. 🔨 `Dropout(0.25)` in [`train.py`](../train.py).
- **Encoder / decoder**: an encoder compresses an input into a representation; a decoder turns a representation back into an output. 🔨 The [VAE](../autoencoder/train_vae.py) has both.
- **Latent space / latent vector**: the compressed, hidden representation inside a model. 🔨 The VAE squeezes each digit into a 2-number latent vector.
- **Hidden state**: the values passed between layers (or time steps) inside a network.
- **Universal approximation theorem**: the result that a big enough network can approximate almost any function. It says this is *possible*, not that training will find it.

## 4. How training works

- **Gradient**: the direction and amount each weight should change to reduce the loss most quickly. Picture the slope of a hill.
- **Gradient descent**: repeatedly nudging every weight a little bit downhill on the loss. It's the core of almost all neural network training.
- **Stochastic gradient descent (SGD)**: gradient descent using a random mini-batch each step instead of the whole dataset. Faster, and the noise even helps.
- **Backpropagation (backprop)**: the algorithm that computes the gradient for every weight, working backwards through the network with the chain rule. 🔨 `loss.backward()` in every training script.
- **Forward pass / backward pass**: forward runs the input through the model to get a prediction and a loss; backward computes the gradients.
- **Learning rate**: how big each weight update is. Too high and training explodes; too low and it crawls. 🔨 `LEARNING_RATE = 1e-3` in [`train.py`](../train.py), and a smaller `1e-4` for fine-tuning.
- **Optimizer**: the rule that turns gradients into weight updates. 🔨 `optimizer.step()`.
  - **Momentum**: keeps some of the previous update's direction, like a ball rolling downhill.
  - **Adam**: the most popular optimizer. It adapts the step size for each weight. 🔨 Used by the digit recognizer, VAE and DQN.
  - **AdamW**: Adam with weight decay done properly. The default for transformers. 🔨 Used by the tiny GPT.
  - Others: RMSProp, AdaGrad, Lion, Adafactor, Muon.
- **Learning rate schedule**: changing the learning rate during training.
  - **Warmup**: start small and ramp up, so early updates don't wreck a fresh network.
  - **Cosine decay**: smoothly lower the rate along a cosine curve. 🔨 `CosineAnnealingLR` in [`language_model/train_lm.py`](../language_model/train_lm.py).
- **Convergence**: when the loss stops improving much, so training has settled.
- **Local minimum / saddle point**: places where the loss is flat but not the best possible. In big networks these matter less than people once feared.
- **Vanishing / exploding gradients**: gradients shrinking to nothing, or blowing up, as they pass back through many layers. Residual connections, normalization and good initialization fix most of it.
- **Gradient clipping**: capping how big a gradient can be, to stop one bad batch from causing a huge update.
- **Weight initialization**: how weights are set before training (Xavier/Glorot, He/Kaiming). A bad start can stop a network learning at all.
- **Regularization**: anything that reduces overfitting: dropout, weight decay, data augmentation, early stopping.
- **Weight decay / L2 regularization**: gently shrinking all weights toward zero each step, so the model prefers simpler solutions.
- **L1 regularization**: pushes many weights to exactly zero.
- **Early stopping**: stopping training when the validation score stops improving. 🔨 Similar idea: [`train.py`](../train.py) and [`snake_dqn/train_dqn.py`](../snake_dqn/train_dqn.py) save only the *best* version seen.
- **Checkpoint**: a saved copy of a model's weights (and sometimes the optimizer state) during or after training. 🔨 The `.pt` files.
- **Loss functions**:
  - **Cross-entropy**: the standard loss for classification. It punishes confident wrong answers hard. 🔨 [`train.py`](../train.py) and the tiny GPT.
  - **Mean squared error (MSE)**: the average of (prediction − answer)². Standard for regression.
  - **Mean absolute error (MAE)**: the average of the absolute differences. Less sensitive to outliers.
  - **Huber loss (smooth L1)**: MSE for small errors and MAE for large ones. 🔨 The DQN uses it so one surprising crash doesn't produce a huge update.
  - **Binary cross-entropy**: cross-entropy for yes/no outputs. 🔨 The VAE's rebuild loss.
  - **KL divergence**: measures how different two probability distributions are. 🔨 The VAE's second loss term.
- **Mixed precision (FP16 / BF16)**: doing most of the math with 16-bit numbers instead of 32-bit, so training is about twice as fast and uses less memory.
- **Gradient accumulation**: adding up gradients over several small batches before updating, to imitate a bigger batch on limited memory.
- **Gradient checkpointing**: saving memory by recomputing some values during the backward pass instead of storing them.
- **Training loop**: the forward → loss → backward → step cycle, repeated. Almost every method is a variation on it. 🔨 See [The core training loop](<../AI Training Methods/TRAINING_METHODS.md#the-core-training-loop>).
- **Freezing**: locking some weights so training doesn't change them. 🔨 `requires_grad = False` in [`finetune.py`](../finetune.py).

## 5. Data

- **Synthetic data**: data *generated* (by a program, a simulation or another AI) instead of collected from the real world. Useful when real data is scarce, private or expensive. Many LLMs are now trained partly on text written by other LLMs. 🔨 The Snake agents train only on games they generate themselves.
- **Data augmentation**: making extra training examples by slightly changing existing ones: rotating, shifting, cropping, adding noise. 🔨 `RandomAffine` in [`finetune.py`](../finetune.py) wiggles your drawings so ~90 examples go further.
- **Preprocessing**: cleaning and reshaping raw data into what the model expects. 🔨 `to_mnist_style()` in [`finetune.py`](../finetune.py) crops, resizes and centers your drawings like MNIST digits.
- **Normalization / standardization**: rescaling inputs to a standard range, often mean 0 and spread 1, so training is smoother. 🔨 `Normalize((0.1307,), (0.3081,))`: MNIST's mean and standard deviation.
- **Labeling / annotation**: humans (or models) attaching correct answers to data.
- **Ground truth**: the true, correct answers you compare predictions against.
- **Noisy labels**: labels that are sometimes wrong.
- **Class imbalance**: when some categories have far more examples than others (99 healthy scans for every 1 sick one). Accuracy alone can then be misleading.
- **Distribution**: the overall pattern of the data: what's common and what's rare.
- **Distribution shift / data drift**: when real-world data stops looking like the training data. 🔨 Your crossed 7s vs. MNIST's uncrossed ones: the model was 99% accurate on MNIST but read your 7 as a 2.
- **Out-of-distribution (OOD)**: inputs unlike anything the model saw in training.
- **Data leakage**: test information sneaking into training, so the score looks better than it really is.
- **Benchmark contamination**: when test questions end up in a model's training data, inflating its scores.
- **Deduplication**: removing repeated examples, which otherwise get memorized.
- **Dataset bias**: a dataset that over- or under-represents some groups or cases, so the model learns that skew.
- **Feature engineering**: designing useful inputs by hand. 🔨 The DQN's "how much space can I still reach?" feature, which doubled its score.
- **One-hot encoding**: representing a category as a list of 0s with a single 1, like `[0,0,1,0]` for "category 3".
- **Tokenization**: splitting text into pieces (tokens) for a language model. See [section 6](#6-transformers-and-language-models).
- **Corpus**: a large collection of text used for training.
- **Well-known datasets**: **MNIST** (70,000 handwritten digits) 🔨 used here; **CIFAR-10** (small color photos); **ImageNet** (14 million labeled photos); **Common Crawl** (a huge scrape of the web, used for LLMs).

## 6. Transformers and language models

- **Large language model (LLM)**: a very large transformer trained on huge amounts of text to predict the next token. ChatGPT, Claude, Gemini and Llama are LLMs.
- **Small language model (SLM)**: a smaller LLM, often 1–10 billion parameters, cheap enough to run on a laptop or phone.
- **Foundation model**: a big, general model trained on broad data that can be adapted to many tasks.
- **Token**: the unit of text an LLM reads and writes. Usually a word or part of a word; roughly ¾ of an English word on average. 🔨 The tiny GPT uses single characters as tokens.
- **Tokenizer**: the tool that splits text into tokens and turns them into numbers. 🔨 `encode()` / `decode()` in [`language_model/train_lm.py`](../language_model/train_lm.py).
- **Byte-pair encoding (BPE)**: the common tokenizing method: start from characters and repeatedly merge the most frequent pairs into new tokens.
- **Vocabulary**: the set of all tokens a model knows. 🔨 The tiny GPT's vocabulary is the 65 characters in Shakespeare.
- **Context window / context length**: how many tokens a model can look at at once. 🔨 `BLOCK_SIZE = 128` characters in the tiny GPT; frontier LLMs handle hundreds of thousands of tokens.
- **Attention**: the mechanism that lets each token look at other tokens and decide which ones matter for it. In "the cat sat because *it* was tired", attention links *it* to *cat*.
- **Self-attention**: attention where a sequence attends to itself.
- **Query, key, value (Q, K, V)**: the three vectors attention computes for each token. A query is "what I'm looking for", a key is "what I contain", and a value is "what I pass on if picked".
- **Multi-head attention**: several attention calculations in parallel, each free to track a different pattern. 🔨 `N_HEADS = 4` in the tiny GPT.
- **Causal mask**: blocking each token from seeing tokens *after* it, so the model can't cheat when predicting the next one. 🔨 Used in the tiny GPT.
- **Positional encoding / embedding**: information about *where* each token sits, since attention alone doesn't know word order. 🔨 `pos_embed` in the tiny GPT; many modern LLMs use **RoPE** (rotary position embedding) instead.
- **Transformer block**: attention plus a small MLP, each wrapped with normalization and a residual connection. Models stack many of these. 🔨 `N_LAYERS = 4` in the tiny GPT.
- **Decoder-only model**: a transformer that generates text left to right, like GPT. 🔨 The tiny GPT.
- **Encoder-only model**: a transformer that reads a whole text at once to understand it, like BERT. Used for search and classification.
- **Encoder-decoder model**: reads an input with one stack and writes an output with another, like T5 or the original translation transformer.
- **Next-token prediction**: the training task for LLMs: given the text so far, predict what comes next. 🔨 Exactly what the tiny GPT trains on.
- **Autoregressive generation**: producing output one token at a time, feeding each new token back in as input. 🔨 `generate()` in the tiny GPT.
- **Perplexity**: a score for how "surprised" a language model is by real text. Lower is better.
- **Decoding / sampling**: choosing the next token from the model's probabilities.
  - **Greedy decoding**: always pick the most likely token. Often repetitive. 🔨 `greedy()` in [`language_model/train_lm.py`](../language_model/train_lm.py).
  - **Temperature**: below 1 makes output safer and more predictable; above 1 makes it more random and creative. 🔨 `TEMPERATURE = 0.8` in [`language_model/generate.py`](../language_model/generate.py).
  - **Top-k sampling**: only choose among the k most likely tokens. 🔨 `TOP_K = 20` in the same file.
  - **Top-p (nucleus) sampling**: only choose among the smallest set of tokens whose probabilities add up to p (like 0.9).
  - **Beam search**: keep several candidate sequences at once and pick the best overall.
- **Prompt**: the text you give an LLM. **System prompt**: instructions set by the app or developer that shape how the model behaves.
- **Prompt engineering**: writing prompts carefully to get better outputs.
- **Chain of thought (CoT)**: having the model reason step by step before answering, which improves accuracy on hard problems.
- **Reasoning model**: an LLM trained (usually with RL) to think at length before answering. See [RLVR](#7-how-llms-are-trained).
- **Hallucination**: when a model confidently states something false or made up. 🔨 The tiny GPT invents Shakespeare-sounding words that don't exist.
- **KV cache**: storing attention keys and values from earlier tokens so generation doesn't recompute them. A big speed-up.
- **Retrieval-augmented generation (RAG)**: looking up relevant documents and putting them in the prompt, so the model answers from real sources.
- **Vector database**: a database that stores embeddings and finds the most similar ones quickly. Used for RAG and semantic search.
- **Semantic search**: searching by meaning (comparing embeddings) rather than exact keywords.
- **Tool use / function calling**: an LLM calling outside tools (a calculator, web search, code) and using the results.
- **AI agent**: an LLM that works in a loop: plan, act with tools, observe the results, and repeat until a task is done.
- **Multimodal model**: a model that handles more than one kind of data, such as text plus images or audio.
- **Vision transformer (ViT)**: a transformer for images that treats patches of the image like tokens.
- **Mixture of experts (MoE)**: a model with many "expert" sub-networks where only a few run for each token. Lots of parameters, but each token costs relatively little compute.
- **Scaling laws**: the finding that loss falls predictably as you increase model size, data and compute.
- **Emergent abilities**: skills that seem to appear only once models get big enough. Researchers still debate how sudden they really are.
- **Open weights vs. closed model**: open-weights models (like Llama) publish their trained weights for anyone to download; closed models (like GPT-4) are only reachable through an API.

## 7. How LLMs are trained

- **Pretraining**: the first, giant training phase: next-token prediction on trillions of tokens of text. It produces a *base model* that knows a lot but doesn't follow instructions well. 🔨 The tiny GPT is pretraining at a tiny scale.
- **Base model**: a model after pretraining only. It continues text rather than answering questions.
- **Post-training**: everything after pretraining that turns a base model into a helpful assistant: SFT, preference tuning, RL.
- **Supervised fine-tuning (SFT)**: training on example conversations (prompt → good answer) written or checked by people.
- **Instruction tuning**: SFT on many different instructions, so the model learns to follow requests in general.
- **Alignment**: making a model's behavior match what people intend: helpful, honest and harmless.
- **RLHF (reinforcement learning from human feedback)**: people rank several model answers, a *reward model* learns to predict those rankings, then RL (usually PPO) trains the LLM to get high reward.
- **Reward model**: a model that scores how good a response is, trained on human preferences.
- **Preference data**: pairs of responses where a person (or AI) marked which one is better.
- **PPO (proximal policy optimization)**: the RL algorithm classically used in RLHF. It limits how far each update can change the model. Also used for games and robots.
- **DPO (direct preference optimization)**: learns straight from preference pairs, with no separate reward model or RL loop. Simpler than RLHF. Relatives: IPO, KTO, ORPO, SimPO.
- **RLAIF (RL from AI feedback)**: like RLHF, but an AI gives the feedback instead of people.
- **Constitutional AI**: Anthropic's method where the model critiques and revises its own answers using a written list of principles (a "constitution").
- **RLVR (RL with verifiable rewards)**: RL where the reward is checked automatically, like "is the math answer correct?" or "do the tests pass?". This is how reasoning models learn to think through problems.
- **GRPO (group relative policy optimization)**: an RL algorithm, made popular by DeepSeek, that scores each answer against a group of the model's own answers to the same question, so it needs no separate value model.
- **Reward hacking**: when a model finds a loophole that earns reward without doing the intended task. See [section 10](#10-reinforcement-learning-rl).
- **Knowledge distillation**: training a small "student" model to imitate a large "teacher" model's outputs. It gives a smaller, faster model that keeps much of the teacher's skill.
- **Synthetic data (for LLMs)**: training data written by other models, such as practice problems with worked answers. Widely used in post-training. See [section 5](#5-data).
- **Model collapse**: quality loss that can happen when models train on too much AI-generated data and too little real data, losing the rare and varied parts of the original distribution.
- **Data mixture**: the recipe of sources (web, code, books, math) and how much of each goes into training.
- **Continued pretraining**: more pretraining on new data (like medical texts) to add domain knowledge.
- **Model merging**: combining the weights of several fine-tuned models into one, without more training.

## 8. Fine-tuning and PEFT

- **Fine-tuning**: taking a pretrained model and training it a bit more on your own data. Much cheaper than training from scratch. 🔨 [`finetune.py`](../finetune.py) fine-tunes the digit model on 123 of your drawings.
- **Full fine-tuning**: updating *all* the weights. Powerful, but needs a lot of memory for big models.
- **Freezing layers**: keeping the early layers fixed and training only the last ones. 🔨 [`finetune.py`](../finetune.py) freezes the convolution layers and trains only the classifier.
- **PEFT (parameter-efficient fine-tuning)**: methods that fine-tune a large model by training only a *tiny fraction* of its weights, often under 1%. Training uses far less memory, and each new task is just a small file.
- **LoRA (low-rank adaptation)**: the most popular PEFT method. The original weights stay frozen, and small trainable matrices are added beside them; their product is a low-rank "update" to the frozen weights. A LoRA for a big model can be just a few megabytes.
- **QLoRA**: LoRA on top of a model *quantized* to 4-bit, which lets you fine-tune big models on one consumer GPU.
- **Adapters**: small trainable layers inserted inside a frozen model.
- **Prompt tuning / prefix tuning**: learning a few special "soft prompt" vectors placed in front of the input, while the model itself stays frozen.
- **Rank (in LoRA)**: the size of the LoRA matrices. Higher rank means more capacity but more parameters (commonly 4–64).
- **Catastrophic forgetting**: fine-tuning can make a model forget old skills. Freezing layers, PEFT and low learning rates all reduce it. 🔨 Measured in [`finetune.py`](../finetune.py): MNIST accuracy only fell from 99.29% to 98.63%.
- **Domain adaptation**: adjusting a model to a new kind of data, such as from photos to X-rays, or from MNIST to your handwriting.
- **Quantization**: storing weights with fewer bits (like 8-bit or 4-bit instead of 32-bit), so the model is smaller and faster with little quality loss. Formats include **INT8**, **INT4**, **GPTQ**, **AWQ** and **GGUF**.
- **Pruning**: removing weights or neurons that barely matter, to shrink a model.

## 9. Generative models

- **Generative model**: a model that *creates* new data (images, text, audio) that looks like its training data. 🔨 The VAE draws new digits; the tiny GPT writes new "Shakespeare".
- **Discriminative model**: a model that *tells things apart* (classifies) instead of creating. 🔨 The digit recognizer.
- **Autoencoder**: a network that compresses its input into a small code, then rebuilds the input from that code. It learns what matters by being forced through a bottleneck.
- **Variational autoencoder (VAE)**: an autoencoder whose code is a *probability cloud* rather than one point, which makes the latent space smooth enough to sample brand-new examples from. 🔨 [`autoencoder/train_vae.py`](../autoencoder/train_vae.py).
- **Reparameterization trick**: writing a random sample as `mean + noise × spread`, so gradients can still flow through it. 🔨 `z = mu + torch.randn_like(mu) * ...` in the VAE.
- **β-VAE**: a VAE with a weight β on its KL term, trading sharp rebuilds against a tidier latent space. 🔨 `BETA` in the VAE.
- **Reconstruction**: the rebuilt output of an autoencoder, compared with the original.
- **Interpolation**: moving smoothly between two points in latent space to morph one output into another. 🔨 The VAE's latent map shows digits morphing into each other.
- **GAN (generative adversarial network)**: two networks in competition. A *generator* makes fakes and a *discriminator* tries to spot them, and both improve until the fakes look real.
- **Mode collapse**: when a GAN keeps producing only a few kinds of output.
- **Diffusion model**: a model that learns to remove noise step by step. To generate, it starts from pure noise and "denoises" it into an image. Stable Diffusion, DALL·E 3 and Midjourney work this way.
- **Denoising**: removing noise from data, the core skill of diffusion models.
- **Latent diffusion**: running diffusion in a compressed latent space instead of on raw pixels, which is much faster.
- **Flow matching**: a newer relative of diffusion that learns a smooth path from noise to data, often needing fewer steps.
- **Text-to-image / text-to-video / text-to-speech**: generative models conditioned on a text description.
- **Conditioning**: giving a generative model extra input (a text prompt, a class label) to steer what it makes.
- **Classifier-free guidance (CFG)**: a diffusion trick for making images follow the prompt more strongly. The "guidance scale" sets how strongly.

## 10. Reinforcement learning (RL)

- **Reinforcement learning (RL)**: an agent learns by acting, getting rewards or penalties, and adjusting so that reward adds up over time. 🔨 [`snake_rl/`](../snake_rl/) and [`snake_dqn/`](../snake_dqn/).
- **Agent**: the learner that takes actions. 🔨 The snake's brain.
- **Environment**: the world the agent acts in. 🔨 The Snake game ([`snake_rl/snake.py`](../snake_rl/snake.py)).
- **State / observation**: what the agent knows about the situation right now. 🔨 11 yes/no facts for the Q-table snake; 14 numbers for the DQN.
- **Action**: a choice the agent can make. 🔨 Go straight, turn left or turn right.
- **Reward**: the score signal after an action. 🔨 +10 for an apple, −10 for dying.
- **Return**: the total (discounted) reward from now until the end of an episode.
- **Episode**: one full run, from start to finish. 🔨 One game of Snake.
- **Policy**: the agent's strategy, a rule for which action to take in each state.
- **Value function**: how good it is to be in a state, counting future rewards.
- **Q-value, Q(s, a)**: how good it is to take action *a* in state *s*, counting future rewards.
- **Q-table**: a lookup table of Q-values for every state and action. 🔨 [`snake_rl/q_table.json`](../snake_rl/q_table.json). You can open it and read it.
- **Q-learning**: learning Q-values by updating each one toward `reward + γ × best future Q`. 🔨 `q_update()` in [`snake_rl/train_q.py`](../snake_rl/train_q.py).
- **Bellman equation**: the rule behind Q-learning: a state's value equals the reward now plus the discounted value of where you end up.
- **Discount factor (γ, gamma)**: how much future rewards count compared with immediate ones (0 = only now, close to 1 = far future). 🔨 `GAMMA = 0.9` in [`snake_rl/train_q.py`](../snake_rl/train_q.py).
- **Exploration vs. exploitation**: trying new actions to discover better ones vs. using the best action known so far.
- **Epsilon-greedy (ε)**: explore at random with probability ε, otherwise take the best known move; ε usually shrinks over training. 🔨 `choose_action()`, with ε going from 1.0 down to 0.01.
- **Deep Q-network (DQN)**: Q-learning with a neural network estimating Q-values instead of a table, so it works when there are too many states to list. 🔨 [`snake_dqn/train_dqn.py`](../snake_dqn/train_dqn.py).
- **Experience replay (replay buffer)**: storing past moves and training on random batches of them, which breaks up correlations between consecutive moves. 🔨 50,000 moves in the DQN.
- **Target network**: a frozen copy of the Q-network used to compute training targets, synced every so often, which keeps learning stable. 🔨 `SYNC_EVERY = 1000` in the DQN.
- **Tabular vs. deep RL**: tabular methods store a value for every state; deep RL uses a network that *generalizes* between states. 🔨 Compare [`snake_rl/`](../snake_rl/) and [`snake_dqn/`](../snake_dqn/).
- **On-policy vs. off-policy**: on-policy methods learn only from the current policy's experience (PPO); off-policy methods can learn from older or other experience (Q-learning, DQN).
- **Policy gradient**: methods that adjust the policy directly toward actions that led to high return. REINFORCE is the simplest.
- **Actor-critic**: two parts, an *actor* that picks actions and a *critic* that judges them. A2C, A3C and SAC are examples.
- **PPO (proximal policy optimization)**: a stable, popular policy-gradient method. Used for robots, games and RLHF.
- **Advantage**: how much better an action was than average in that state.
- **Model-based vs. model-free RL**: model-based agents learn how the world works and plan with it (MuZero); model-free agents learn what to do directly (Q-learning).
- **Monte Carlo tree search (MCTS)**: planning by simulating many possible futures as a search tree. Used by AlphaGo.
- **Self-play**: an agent improving by playing against copies of itself. Used by AlphaZero.
- **Multi-agent RL**: several agents learning in the same environment, cooperating or competing.
- **Reward shaping**: adding extra in-between rewards to guide learning. 🔨 A suggested experiment in [`snake_rl/NOTES.md`](../snake_rl/NOTES.md): a small penalty every step.
- **Sparse reward**: rewards that arrive rarely, which makes learning hard.
- **Reward hacking**: the agent exploits a loophole in the reward instead of doing what you meant.
- **Credit assignment**: working out *which* earlier actions deserve credit for a later reward.
- **Sim-to-real**: training in a simulation, then transferring to the real world (common in robotics).
- **Gym / Gymnasium**: a standard Python library of RL environments.

## 11. Evaluation

- **Accuracy**: the fraction of predictions that are right. 🔨 99.29% for the digit recognizer.
- **Precision**: of everything the model flagged as positive, how much really was.
- **Recall**: of everything that really was positive, how much the model caught.
- **F1 score**: one number that balances precision and recall.
- **Confusion matrix**: a table of what each class was predicted as, which shows *which* mistakes happen (like 7s read as 2s).
- **ROC curve / AUC**: shows the trade-off between catching positives and raising false alarms at different thresholds. AUC sums it up (1.0 is perfect, 0.5 is guessing).
- **Confidence / probability**: how sure the model says it is. Confidence isn't correctness. 🔨 The digit model was 100% "sure" your crossed 7 was a 2.
- **Calibration**: whether a model's confidence matches how often it's right (90% confident should mean right about 90% of the time).
- **Cross-validation**: splitting the data several ways and averaging the scores, for a more reliable estimate on small datasets.
- **Held-out set**: data kept out of training and used only to test. 🔨 25% of your drawings in [`finetune.py`](../finetune.py).
- **Benchmark**: a standard test set used to compare models (MMLU, GSM8K, SWE-bench, HumanEval for LLMs).
- **Eval (evaluation)**: a test of how well a model does at something. In LLM work, "evals" often means custom test suites.
- **Ablation study**: removing one part to see how much it mattered. 🔨 A suggested experiment in [`snake_dqn/NOTES.md`](../snake_dqn/NOTES.md): drop the flood-fill features.
- **Learning curve**: a chart of score or loss over training time. 🔨 The tables in each project's `NOTES.md`.
- **Validation loss**: loss on the validation set. If it rises while training loss keeps falling, the model is overfitting. 🔨 The tiny GPT reports it (1.649).
- **Bias-variance trade-off**: simple models make consistent but systematic errors (high bias, underfitting); complex models fit noise (high variance, overfitting). Good models balance the two.
- **Statistical significance / variance**: small test sets give noisy scores, so results can change by chance. 🔨 The handwriting test set has only 30 drawings, so treat 90.0% → 93.3% with care.
- **Regression test (for models)**: checking a model still hits its reported numbers after changes. 🔨 [`tests/`](../tests/) does exactly this on every push.

## 12. Hardware, scale and deployment

- **GPU (graphics processing unit)**: a chip that does thousands of calculations in parallel, ideal for neural networks. 🔨 The scripts use one automatically if you have it.
- **CUDA**: NVIDIA's platform for running code on its GPUs. 🔨 `"cuda"` in the device line of every training script.
- **MPS (Metal Performance Shaders)**: Apple's way to use the GPU in M-series Macs. 🔨 `"mps"` in the same line.
- **TPU / NPU**: chips built specifically for neural networks (Google's TPUs; the NPUs in phones and laptops).
- **CPU**: the computer's main processor. Slower for big networks, but fine for small ones. 🔨 The DQN trains on CPU in about 3 minutes.
- **VRAM**: the GPU's own memory. It limits how big a model and batch you can fit.
- **FLOPs**: floating-point operations, the standard measure of how much compute training or inference takes.
- **Compute**: the total processing power spent on training, often the biggest cost of training an LLM.
- **Parameter count**: model size, written like 7B (7 billion) or 70B. 🔨 The tiny GPT has about 800k (0.0008B).
- **Distributed training**: spreading training across many GPUs or machines.
  - **Data parallelism**: each GPU holds a copy of the model and trains on different data.
  - **Model / tensor / pipeline parallelism**: splitting one model across GPUs when it's too big for one.
  - **FSDP / ZeRO**: techniques that split the model's weights and optimizer state across GPUs to save memory.
- **Latency / throughput**: latency is how long one prediction takes; throughput is how many predictions per second.
- **Deployment / serving**: running a model somewhere people can use it, like an app or an API. 🔨 The [live demo](https://kiddo4993.github.io/train-ai-from-scratch/) is deployed with GitHub Pages.
- **Edge / on-device AI**: running models on phones, laptops or gadgets instead of in the cloud. 🔨 The live demo runs fully on your device.
- **ONNX**: a standard file format for moving models between frameworks and runtimes.
- **API**: a way for programs to talk to each other. Many LLMs are used through a paid API.
- **MLOps**: practices for training, deploying, monitoring and updating models reliably, like DevOps for ML.
- **Framework**: a library for building and training models. **PyTorch** 🔨 used here, plus TensorFlow, JAX and Keras. **scikit-learn** is the standard for classical ML.
- **Hugging Face**: a hub for sharing models and datasets, and the home of the popular `transformers` and `peft` libraries.
- **Jupyter / Colab notebook**: an interactive document mixing code, results and notes. Colab runs notebooks free in the cloud. 🔨 [`notebooks/`](../notebooks/).

## 13. Safety and responsible AI

- **AI safety**: making sure AI systems behave as intended and don't cause harm, now and as they get more capable.
- **Bias (fairness)**: when a model treats groups of people unfairly, usually because of skewed training data.
- **Explainability / interpretability**: understanding *why* a model made a decision, or what its internals represent. **Mechanistic interpretability** tries to reverse-engineer the actual circuits inside a network.
- **Red teaming**: deliberately trying to make a model misbehave, to find problems before release.
- **Jailbreak**: a prompt crafted to get around a model's safety rules.
- **Prompt injection**: hidden instructions in content a model reads (like a web page) that try to hijack what it does.
- **Adversarial example**: an input changed slightly, often invisibly to people, so the model gets it wrong.
- **Guardrails**: rules or filters around a model that block unsafe inputs or outputs.
- **Sycophancy**: a model telling users what they want to hear instead of what's true.
- **Data privacy**: protecting the personal information in training data and prompts.
- **Watermarking**: hiding a detectable signal in AI-generated content so it can be identified later.
- **Model card**: a document describing what a model is for, how it was trained, how well it does and its limitations.

---

**Missing a term, or found one explained unclearly?** [Open an issue](https://github.com/Kiddo4993/train-ai-from-scratch/issues/new/choose) or send a pull request. See [CONTRIBUTING.md](../CONTRIBUTING.md). Adding terms is a great first contribution.
