# searchloop

searchloop is a planning agent that does MCTS tree search over tool calls, benchmarked against a greedy ReAct baseline.

## LLM client

OpenAI and Anthropic action proposers are available behind the shared `Proposer` protocol.

## MCTS

Search uses a shaped leaf value, true reward plus an evidence bonus, while evaluation and benchmark reporting use the true environment reward.

- [ ] scaffold
- [x] tool harness
- [x] state model
- [x] llm client
- [x] greedy baseline
- [x] benchmark
- [x] mcts selection
- [x] mcts rollouts
- [x] proposal caching
- [x] progressive widening
- [x] tuning
- [x] trace collection
- [x] learned value head
- [x] ablations
