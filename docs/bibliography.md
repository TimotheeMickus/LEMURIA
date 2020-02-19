---
    title: 'Bibliography'
    geometry: 'margin=1.5cm'
    classoption: 10pt
    colorlinks: true
...

## EmeCom19

[website](https://sites.google.com/view/emecom2019/home). Most accepted papers are available directly [from this link](https://sites.google.com/view/emecom2019/accepted-papers).

**Useful/ relevant papers**

 - Playing log(N)-Questions over Sentences. *Variation on emergent communication games based on question answering*
 - Enhancing Communication Learning through Empathic Prediction. *Augment an emergent communication task with an auxiliary predictive task, consisting in predicting the other player's hidden state*
 - Focus on What’s Informative and Ignore What’s not: Communication Strategies in a Referential Game. *Study the lexicon by-product of communication games, and suggest to adopt a non-uniform distribution over categories. Use of interesting metrics (mutual information) to mesure the *
 - Avoiding hashing and encouraging visual semantics in referential emergent language games. *Studies factors necessary to encode visual semantics rather than trivial low-level pixel properties*
 - Enhance the Compositionality of Emergent Language by Iterated Learning. *Uses iterated learning to learn communication*
 - To Populate Is To Regulate. *Advocates in favor of population-based signaling games.*
 - On Emergent Communication in Competitive Multi-Agent Teams. [paper](https://www.semanticscholar.org/paper/On-Emergent-Communication-in-Competitive-Teams-Liang-Chen/e1d308595eaa253574cda03e7f5fdcff38abb42e). *Advocates in favor of competition as an environmental pressure for learning composition*
 - Emergence of Pragmatics from Referential Game between Theory of Mind Agents. *Linguistically informed design*
 - Developmentally motivated emergence of compositional communication via template transfer. *Pretraining of communication protocols*


**Perhaps useful for perspective**

 - Biology and Compositionality: Empirical Considerations for Emergent-Communication Protocols. *Discusses studies of compositionality in emergent communication from the perspective of biology*

 **Less relevant**

 - Learning Autocomplete Systems as a Communication Game. *Treats sentence completion as a communication game*
 - The Emergence of Compositional Languages for Numeric Concepts Through Iterated Learning in Neural Agents. [paper](https://arxiv.org/abs/1910.05291). *Specifically focuses on signaling games based on distinct targets composed of similar objects in different quantities*
 - Unnatural Language Processing: Bridging the Gap Between Synthetic and Natural Language Data. [paper](https://alanamarzoev.github.io/pdfs/unnatural_language.pdf). *Studies synthetic data*
 - EGG: a toolkit for research on Emergence of lanGuage in Games. *Toolkit.*
 - Improving Policies via Search in Cooperative Partially Observable Games. *Game RL*
 - Emergent Communication with World Models. *Mostly about language modeling in RL settings.*


**404 paper not found**

 - Embodied Multi-Agent Learning in Minecraft.

## EmeCom18

[website](https://sites.google.com/site/emecom2018/)

**Useful/ relevant papers**

 - Intrinsic Social Motivation via Causal Influence in Multi-Agent RL. [paper](https://deepmind.com/research/publications/intrinsic-social-motivation-causal-influence-multi-agent-rl), [more recent version](https://arxiv.org/abs/1810.08647v4)
 - How not to measure the emergence of communication. [paper](https://arxiv.org/abs/1903.05168). *Agent population learning to cooperate through language in gridworlds; proposes to add a term to the loss to reward agents whose actions changed the action of other agents in the population ('social influence'). Single symbol per timestep. Uses mutual information to measure the correlation between the emission of a symbol and subsequent actions by other agents, as weel as a consistency metric to evaluate how often a symbol is matched with a specific action.*
 - Incremental Pragmatics and Emergent Communication. [paper](https://www.semanticscholar.org/paper/Incremental-Pragmatics-and-Emergent-Communication-Tomlin-Pavlick/6f899e069ed79860ccb3baaa5f9bae825441258a). *Response to 'Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog.' (cf. infra). Suggest that groundedness can arise from a more structured curriculum of learning.*
 - Emergence of Communication in an Interactive World with Consistent Speakers. [paper](https://arxiv.org/abs/1809.00549).
 - Easy-to-Teach Language Emergence from Multi-Agent Communication Games. [paper](https://arxiv.org/abs/1906.02403).

**Less relevant**

 - Countering Language Drift via Grounding. [paper](https://arxiv.org/abs/1909.04499). *Interesting paper overall. Suggests that scalar-based rewards are not enough to produce human-like language, and includes language-based rewards (eg. likelihood of producing a given token). However, this paper is mostly concerned with finetuning existing agents (eg. translation models)/ mimicking an existing language (eg. French/English) and does not addresses the emergence of language ex nihilo.*
 - Seq2Seq Mimic Games: A Signaling Perspective. [paper](https://arxiv.org/abs/1811.06564). *Interesting paper overall. Adversarial setup for emergence of language: two Answering agents ('red' and 'blue') compete for the reward handed out by an Interrogator agent, the Interrogator has to distinguish 'blue' from 'red' to be rewarded. Cons: asymetrical setup with respect to Answering agents; moreover no explicit grounding; no discussion of the emergent language*
 - Training an Interactive Helper. [paper](https://arxiv.org/abs/1906.10165).
 - Learning When to Communicate at Scale in Multiagent Cooperative and Competitive Tasks. [paper](https://openreview.net/forum?id=rye7knCqK7).
 - Paying Attention to Function Words. [paper](https://arxiv.org/abs/1909.11060).
 - Learning to Activate Relay Nodes: Deep Reinforcement Learning Approach. [paper](https://arxiv.org/abs/1811.09759).
 - Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning. [paper](https://arxiv.org/abs/1811.01458).

**404 paper not found**

 - Emergent communication via model-based policy transmission.
 - First steps to understanding how symbolic communication arises.
 - Selective Emergent Communication With Partially Aligned Agents.
 - Emergence of (Grounded) Compositional Language - A (Short) Review.

## EmeCom17

[website](https://sites.google.com/site/emecom2017/)

**Useful/ relevant papers**

 - Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog. [paper](https://arxiv.org/abs/1706.08502). *Addresses the evaluation of compositionality and groundedness in agent-generated languages. No multi-symbol production*
 - Multi-Agent Cooperation and the Emergence of (Natural) Language. [paper](https://arxiv.org/abs/1612.07182). *Referential game with single-symbol messages*
 - Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols. [paper](https://arxiv.org/abs/1705.11192). *Original paper on discrete sequential message exchange in emergent communication. Work conducted on real images (MSCOCO), suggests that Gumbel-softmax is better than RL for this task*
 - Emergent Communication through Negotiation. [paper](https://arxiv.org/abs/1804.03980).
 - Analyzing Language Learned by an Active Question Answering Agent. [paper](https://arxiv.org/abs/1801.07537).
 - Multi-Agent Compositional Communication Learning from Raw Visual Input. [paper](https://www.semanticscholar.org/paper/Multi-Agent-Compositional-Communication-Learning-Choi-Lazaridou/08bbcf6f753a4889f57cede3b0bdedb56024cc03), [more recent version](https://openreview.net/forum?id=rknt2Be0-). *Same paper as 'compositional obverter', cf. infra.*

**Perhaps useful for perspective**

 - On The Evolutionary Origin of Symbolic Communication. [paper](https://www.nature.com/articles/srep34615).

**Less relevant**

 - Modeling Opponent Hidden States in Deep Reinforcement Learning. [more recent version](https://arxiv.org/abs/1802.09640).
 - Vehicle Communication Strategies for Simulated Highway Driving. [paper](https://www.semanticscholar.org/paper/Vehicle-Communication-Strategies-for-Simulated-Resnick-Kulikov/05a1c483e68d8af0a4133b902e5c8f62a656e65e).
 - MACE: Structured Exploration via Deep Hierarchical Coordination. [paper](https://openreview.net/forum?id=HyunpgbR-)
 - A two dimensional decomposition approach for matrix completion through gossip. [paper](https://arxiv.org/abs/1711.07684).
 - Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games. [paper](https://arxiv.org/abs/1703.10069).

**404 paper not found**

 - Multi-agent Communication by Bi-directional Recurrent Neural Networks.

## Other papers

 - Emergence and Artificial Life. [paper](https://www.researchgate.net/publication/4048749_Emergence_and_artificial_life).
 - Synthetic Ethology: An Approach to the Study of Communication. [paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.6635).
 - Spontaneous Evolution of Linguistic Structure—An Iterated Learning Model of the Emergence of Regularity and Irregularity. [paper](https://ieeexplore.ieee.org/document/918430 ).
 - Mastering emergent language: learning to guide in simulated navigation. [paper](https://arxiv.org/abs/1908.05135).
 - Entropy Minimization In Emergent Languages. [paper](https://arxiv.org/abs/1905.13687).
 - Anti-efficient encoding in emergent communication. [paper](https://arxiv.org/abs/1905.12561).
 - Word-order biases in deep-agent emergent communication. [paper](https://arxiv.org/abs/1905.12330).
 - Miss Tools and Mr Fruit: Emergent communication in agents learning about object affordances. [paper](https://arxiv.org/abs/1905.11871).
 - On Voting Strategies and Emergent Communication. [paper](https://arxiv.org/abs/1902.06897).
 - Emergent Linguistic Phenomena in Multi-Agent Communication Games. [paper](https://arxiv.org/abs/1901.08706).
 - How agents see things: On visual representations in an emergent language game. [paper](https://arxiv.org/abs/1808.10696).
 - Talk the Walk: Navigating New York City through Grounded Dialogue. [paper](https://arxiv.org/abs/1807.03367).
 - Encoding Spatial Relations from Natural Language. [paper](https://arxiv.org/abs/1807.01670).
 - Emergence of Linguistic Communication from Referential Games with Symbolic and Pixel Input. [paper](https://arxiv.org/abs/1804.03984). *Well, duh.*
 - Exploring Structural Inductive Biases in Emergent Communication [paper](https://arxiv.org/abs/2002.01335).
 - On the interaction between supervision and self-play in emergent communication. [paper](https://arxiv.org/abs/2002.01093).
 - Biases for Emergent Communication in Multi-agent Reinforcement Learning. [paper](https://arxiv.org/abs/1912.05676).
 - Learning to Request Guidance in Emergent Communication. [paper](https://arxiv.org/abs/1912.05525).
 - Capacity, Bandwidth, and Compositionality in Emergent Language Learning. [paper](https://arxiv.org/abs/1910.11424). *Expecially dedicated to analysis of emergent languages. Defines useful metrics. Has a VAE setup that somewhat hinders the exploitability of the proposed results.*
 - Emergence of Writing Systems Through Multi-Agent Cooperation. [paper](https://arxiv.org/abs/1910.00741).
 - Learning to Communicate with Deep Multi-Agent Reinforcement Learning. [paper](https://arxiv.org/abs/1605.06676).
 - Compositional Obverter Communication Learning from Raw Visual Input. [paper](https://openreview.net/forum?id=rknt2Be0-). *Manual analysis of the grammar of generated productions through grammar*
 - Emergence of Grounded Compositional Language in Multi-Agent Populations. [paper](https://www.semanticscholar.org/paper/Emergence-of-Grounded-Compositional-Language-in-Mordatch-Abbeel/5d2f5c2dc11c18c0d45203e2b980fe375a56d774). *Worth noting. fairly complex setup, where agents have different goals, and language is one way (though not the only way) to transmit information to other agent and achieve said goals. Each agent only emits one symbol per timestep, but multiple timesteps per episode allow for compositional grounded language to emerge*
 - Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning. [paper](https://arxiv.org/abs/1703.06585).*Initial proposal for the A-bot and Q-bot language game. Also tackles description of actual images in English; involves pretrqining.*
