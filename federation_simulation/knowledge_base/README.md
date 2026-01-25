# Knowledge Base Storage Directory

This directory stores persistent knowledge bases for the Federated Learning simulation.

## Structure

```
knowledge_base/
├── global_knowledge_base.json    # Server's aggregated knowledge
└── agents/
    ├── Agent_A_knowledge_base.json
    ├── Agent_B_knowledge_base.json
    └── ...
```

## Global Knowledge Base Format

```json
{
  "updates": [
    {
      "agent_id": "Agent_A",
      "attack_signature": [0.1, 0.2, ...],
      "mitigation_policy": "...",
      "attack_category": "Fuzzers",
      "received_at": "2026-01-25T...",
      "is_zero_day": true
    }
  ],
  "zero_day_count": 10,
  "total_updates": 150,
  "last_saved": "2026-01-25T..."
}
```

## Local Agent Knowledge Base Format

```json
{
  "agent_id": "Agent_A",
  "known_attacks": ["Fuzzers", "DoS", "Reconnaissance"],
  "signature_hashes": ["abc123...", "def456..."],
  "learned_signatures": [
    {
      "attack_category": "Fuzzers",
      "signature_hash": "abc123...",
      "feature_sample": [0.1, 0.2, ...],
      "mitigation_policy": "...",
      "is_zero_day": true,
      "learned_at": "2026-01-25T..."
    }
  ],
  "stats": {...},
  "last_saved": "2026-01-25T..."
}
```

## Notes

- Knowledge bases are automatically saved after each update
- On restart, agents and server will reload their previous knowledge
- Use the `/reset` endpoint on the server to clear the global KB
- Call `agent.clear_local_knowledge()` to clear an agent's local KB
