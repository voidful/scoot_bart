import time

import pfrl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rl.actor import Actor
from rl.env import Env

tokenizer = AutoTokenizer.from_pretrained("voidful/bart_base_cnndm")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/bart_base_cnndm")

model.cuda()
model.eval()

input_sent = """(CNN) -- An American woman died aboard a cruise ship that docked at Rio de Janeiro on Tuesday, the same ship on which 86 passengers previously fell ill, according to the state-run Brazilian news agency, Agencia Brasil. The American tourist died aboard the MS Veendam, owned by cruise operator Holland America. Federal Police told Agencia Brasil that forensic doctors were investigating her death. The ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension, according the agency. The other passengers came down with diarrhea prior to her death during an earlier part of the trip, the ship's doctors said. The Veendam left New York 36 days ago for a South America tour."""
target_sent = """</s>The woman died aboard the MS Veendam, owned by cruise operator Holland America.\nThe ship's doctors told police that the woman was elderly and suffered from diabetes and hypertension.\nAnother passengers came down with diarrhea prior to her death, the ship's officials said.</s>"""

env = Env(model, tokenizer, observation_input=[[input_sent, target_sent]])
actor = Actor(env, model, tokenizer)
agent = actor.agent_ppo(update_interval=10, minibatch_size=1024, epochs=100)

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=50000,
    eval_n_steps=None,
    eval_n_episodes=1,
    train_max_episode_len=4096,
    eval_interval=10000,
    outdir='scoot_model',
)

agent.load("scoot_model/best")

print("rl model")
s = time.time()
pred_sent = actor.predict(input_sent)
e = time.time()
print(pred_sent)
print(tokenizer.decode(pred_sent['predicted']))
print("time", e - s)

print("rl without actor")
obs = env.reset()
s = time.time()
while True:
    obs, reward, done, pred = env.step(0, return_reward=False)
    if done:
        e = time.time()
        print("time", e - s)
        print(tokenizer.decode(pred['predicted']), pred, len(pred['predicted']), )
        break

print("origin model")
s = time.time()
print(tokenizer.batch_decode(
    model.generate(tokenizer(input_sent, return_tensors='pt')['input_ids'].to(model.device), max_length=1024,
                   output_hidden_states=True, num_beams=1)))
e = time.time()
print("time", e - s)
