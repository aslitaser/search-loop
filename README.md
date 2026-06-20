# searchloop

searchloop is a planning agent that does MCTS tree search over tool calls, benchmarked against a greedy ReAct baseline.

## LLM client

OpenAI and Anthropic action proposers are available behind the shared `Proposer` protocol.

- [ ] scaffold
- [x] tool harness
- [x] state model
- [x] llm client
- [x] greedy baseline
- [x] benchmark
- [ ] mcts selection
- [ ] mcts rollouts
- [ ] progressive widening
- [ ] tuning
- [ ] trace collection
- [ ] learned value head
- [ ] ablations
