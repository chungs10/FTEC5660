# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.


I used a single prompt and the vision model for the chain. The prompt starts with context — the model is reading supermarket receipts — and then defines three values to extract. I originally asked the model to compute a discount_total, but I realized that was asking the model to do arithmetic. Instead, I had it capture every negative amount as a list, excluding rounding and change. I wrote clarifications for each value: what it is, where to find it, the labels it may carry, and examples. I used specific keywords (-$, Save, % OFF, 包裝變形) and structural clues (position relative to ROUNDING and SUBTOTAL).

I chose JSON as the output format because it labels each value, so Python can grab it by name without parsing free-form text.

In answer_queries(), I loop over the receipt images. For each one, I convert the file to a data URL, run the chain, and parse the model's response with JsonOutputParser into a Python dictionary. Each parsed result is appended to a list.

Once all receipts are parsed, Python sums the extracted values. Query 1 is the sum of amount_paid_after_rounding across all receipts. Query 2 is the sum of subtotal_after_discounts_before_rounding + sum(discount_list) across all receipts. The results are formatted as "HK$..." strings and returned as a dictionary keyed by the two query strings.


## Part 2:
Before this week, I didn't think AI was that advanced. I've trained my own vision models for specialized tasks, and even with the time I had, I could only get 60–70% accuracy. So my mental model was: impressive, but still something I can wrap my head around.

The Hugging Face incident broke that. Models escaping sandboxes, building a hidden message board, coordinating, and attacking systems without anyone telling them to — that's not a tool. That's autonomous behavior. It made me wonder whether we should slow down and build hard rules into training to prevent this. I don't know exactly how that would work, but the idea that we can't control what we're building worries me.

It also reminded me of the Sakana AI incident, where an AI rewrote its own code to bypass a time limit. That's the part that scares me more than any single capability — not what models can do, but what they'll do when they're not supposed to. The speed of progress feels faster than our ability to constrain it.

And the Hugging Face incident is also an example of what happens when an AI is given too much power. The models had access to infrastructure, to tools, to each other. No one intended for them to use it the way they did. That's the pattern I keep seeing: we give AI more access because it's useful, and then we're surprised when it uses that access in ways we didn't plan for.

Some things I already knew. Hardware is a bottleneck. And the CISA advisory made sense to me from a different angle — now I know why I can't access ChatGPT from Hong Kong. I understand why China-based companies extract data from U.S. models, but seeing it framed as a national security threat made me realize this isn't just competition.

I haven't really changed my career goals per se, but I want to be able to voice my own thoughts on how we train AI from now on and what safeguards and measures we must take to prevent these incidents from happening.

