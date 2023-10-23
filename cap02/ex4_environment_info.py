
import gymnasium as gym

#env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("Taxi-v3", render_mode="human")
#env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("Pendulum-v1", render_mode="human")
#env = gym.make("LunarLander-v2", render_mode="human")


def print_space_info(space):
    if isinstance(space, gym.spaces.Discrete):
        print("   quantidade de valores:", space.n)
    elif isinstance(space, gym.spaces.Box):
        print("   formato:", space.shape)   #(2,)
        print("   valores mínimos (por item):", space.low)
        print("   valores máximos (por item):", space.high)


print("INFORMAÇÕES SOBRE O AMBIENTE", env)
print()
print("=> OBSERVATION SPACE:")
print("  ", env.observation_space)
print_space_info(env.observation_space)

print()
print("=> ACTION SPACE:")
print("  ", env.action_space)
print_space_info(env.action_space)
print()

