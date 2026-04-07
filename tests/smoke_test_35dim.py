"""Smoke test: verify 35-dim state flows correctly through model inference."""
from postex_agent.core.state import HostState, STATE_DIM
from postex_agent.core.actions import compute_action_mask, ACTION_SPACE_SIZE, Action
from postex_agent.rl.policy_inference import RLPolicy
from postex_agent.environment.state_builder import update_state, update_temporal

print(f"STATE_DIM = {STATE_DIM}")

# 1. Build a realistic state as if parsing real output
state = HostState()
state.os_identified = True
state.user_identified = True

# Simulate CHECK_SUID found 5 exploitable binaries
parsed_suid = {
    "vector_found": True,
    "details": {
        "exploitable_bins": [
            "/usr/bin/find", "/usr/bin/vim", "/usr/bin/python3",
            "/usr/bin/bash", "/usr/bin/nmap",
        ]
    },
}
update_state(state, Action.CHECK_SUID, parsed_suid)

# Simulate CHECK_CREDENTIALS found 2 creds including a private key
parsed_cred = {
    "vector_found": True,
    "details": {
        "credentials": ["db_pass=admin123", "[SSH/RSA private key detected]"],
        "cred_count": 2,
        "cred_quality": 1.0,
    },
}
update_state(state, Action.SEARCH_CREDENTIALS, parsed_cred)

# Simulate a failed EXPLOIT_SUID
parsed_fail = {"vector_found": True, "details": {"is_root": False}}
update_state(state, Action.EXPLOIT_SUID, parsed_fail)

# Update temporal (step 5 of 30, some risk taken)
update_temporal(state, step=5, max_steps=30, cumulative_risk=0.20)

# 2. Convert to vector
vec = state.to_vector()
print(f"Vector shape: {vec.shape}")
print(f"Vector: {vec}")
print()

# 3. Verify new dims are populated
print(f"richness[suid] = {state.richness['suid']:.2f}  (5 bins)")
print(f"richness[cred] = {state.richness['credentials']:.2f}")
print(f"exploit_failures[suid] = {state.exploit_failures['suid']}")
print(f"cred_count = {state.cred_count:.2f}")
print(f"cred_quality = {state.cred_quality:.2f}")
print(f"time_step = {state.time_step:.3f}")
print(f"cumulative_risk = {state.cumulative_risk:.3f}")
print()

# 4. Load model and get prediction
policy = RLPolicy(model_path="artifacts/dqn_model.pt")
action = policy.predict(vec)
top = policy.top_actions(vec, n=3)
print(f"Predicted action: {action.name}")
print("Top-3 actions:")
for a, qv, desc in top:
    print(f"  [{qv:+.3f}] {a.name} -- {desc[:60]}")
print()
print("SMOKE TEST PASSED")
