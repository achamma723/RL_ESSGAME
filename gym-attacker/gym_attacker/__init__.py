from gym.envs.registration import register

register(
    id='ErdosAttack-v0',
    entry_point='gym_attacker.envs:Attacker'
)
