"""
Test an OpenAI API
"""
import requests
import json

prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
    "### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course!"
    " Here are some creative ideas for a 10-year-old's birthday party:\n"
    "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. "
    "Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
    "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. "
    "You can set up different stations with activities like making slime, erupting volcanoes, "
    "or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with "
    "a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, "
    "and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
    "4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity."
    " Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique "
    "masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various "
    "sports and games. Set up different stations for activities like sack races, relay races, basketball shooting"
    ", and obstacle courses. Give out medals or certificates to the participants.\n"
    "6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes,"
    " or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the "
    "kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun "
    "training activities. Set up an obstacle course, have them design their own superhero capes or masks, "
    "and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party "
    "at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with "
    "games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to "
    "the birthday child's interests and preferences. "
    "Have a great celebration!\n### Human: who are you?\n### Assistant:"
)

if __name__ == "__main__":
    rst_flag = 0
    res = requests.post(
        url="http://localhost:8000/v1/completions",
        data=json.dumps(
            {
                "model": "bert-base-cased",
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.5,
            }
        ),
        headers={"Content-Type": "application/json"},
    )

    print(res.text)
    if res.json().get("object") == "error":
        rst_flag += 1
    res = requests.post(
        url="http://localhost:8000/v1/completions",
        data=json.dumps(
            {
                "model": "bloom-560m",
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.5,
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    print(res.text)
    if res.json().get("object") == "error":
        rst_flag += 1
    if rst_flag != 0:
        raise AssertionError("%d case error" % rst_flag)
