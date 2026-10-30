"""
Benchmark Data Generator: 1,000 Realistic Customer Product Reviews
Generates varied, high-quality reviews across sentiment polarities, star ratings,
and product aspects (sound, ANC, battery, mic, comfort, build, software app).
"""

import random
import csv
import os
from datetime import datetime, timedelta

random.seed(42)

PRODUCT_NAME = "ApexSound Pro ANC Wireless Headphones"

PROS_TEMPLATES = [
    ("The active noise cancellation is world-class. It completely cancels out airplane engine hum and office chatter.", "Superb ANC"),
    ("Battery life is insane. I easily got over 38 hours on a single charge with ANC enabled.", "Unbelievable battery life"),
    ("Sound quality is crystal clear with deep punchy bass and well-defined mids and highs.", "Audiophile-grade sound"),
    ("Memory foam earcups are super plush and comfortable. I can wear them for 6 hours straight without ear fatigue.", "Extremely comfortable"),
    ("Seamless multi-point Bluetooth pairing switches instantly between my MacBook and iPhone.", "Flawless multi-device connectivity"),
    ("Fast USB-C charging gave me 5 hours of playback in just 10 minutes of charge.", "Quick charge is a lifesaver"),
    ("Transparency mode sounds completely natural, not robotic or muffled like cheaper competitors.", "Natural transparency mode"),
    ("Premium build materials with sleek brushed aluminum accents and soft leatherette.", "Sturdy premium feel"),
    ("Great EQ customization inside the companion app once you set up custom sound profiles.", "Custom EQ sounds great"),
    ("Physical buttons for volume and track skipping are much better than finicky touch swipe controls.", "Intuitive physical buttons"),
]

CONS_TEMPLATES = [
    ("The mobile companion app constantly loses connection and requires account re-login.", "Frustrating mobile app bugs"),
    ("Microphone quality for phone calls is disappointing. Callers complain that I sound muffled in windy conditions.", "Subpar microphone quality"),
    ("The headband feels slightly heavy after 3 to 4 hours of continuous use.", "A bit heavy for long sessions"),
    ("The plastic hinge on the left ear cup developed an annoying squeak after two months.", "Hinge squeak build issue"),
    ("It does not come with a protective hard-shell case, only a cheap fabric pouch at this price point.", "Disappointing carrying pouch"),
    ("Bluetooth occasionally drops audio for half a second when walking in crowded train stations.", "Occasional Bluetooth stutter in dense areas"),
    ("Ear pads can get rather warm and sweaty during warmer summer days or light walks.", "Ear pads trap heat"),
    ("Lack of aptX Low Latency codec causes noticeable audio sync delay when gaming on PC.", "Noticeable latency for PC gaming"),
    ("Replacement ear cushions are proprietary and ridiculously expensive to purchase separately.", "Costly replacement parts"),
    ("The auto-pause sensor when removing headphones triggers falsely when tilting my head.", "Overly sensitive wear detection"),
]

NEUTRAL_TEMPLATES = [
    ("They are decent headphones overall. Sound is good but not mind-blowing for the retail price.", "Decent but price is steep"),
    ("Good build quality and okay noise cancelling, but nothing that blows Sony or Bose out of the water.", "Average for this category"),
    ("Sound is quite neutral out of the box, you definitely need to boost the EQ if you like bass.", "Needs EQ tuning"),
    ("Battery lasts as advertised. Design is subtle and minimalist, maybe a bit plain.", "Solid workhorse, plain look"),
    ("They work fine for working from home and Zoom calls, though nothing exceptional.", "Acceptable office headphones"),
]

FILLERS_POS = [
    "Overall, strongly recommended to anyone looking for premium travel headphones.",
    "Would definitely buy again without hesitation.",
    "Best purchase I made this year!",
    "Far exceeded my expectations for the price.",
    "A must-have for daily subway commuters.",
]

FILLERS_NEG = [
    "I expected much better build quality and software polish for this price tag.",
    "Customer support was unhelpful when I reported the issue.",
    "Decided to return them and stick with my previous pair.",
    "Look elsewhere if microphone call clarity is your top priority.",
    "Hope the manufacturer fixes these firmware bugs in the next update.",
]

def generate_review(review_idx: int):
    # Distribution: ~55% positive (4-5), ~18% neutral (3), ~27% negative (1-2)
    rand_val = random.random()
    if rand_val < 0.55:
        # Positive review
        rating = random.choice([4, 5])
        pros_pick = random.sample(PROS_TEMPLATES, k=random.randint(2, 3))
        has_minor_con = random.random() < 0.35
        cons_pick = random.sample(CONS_TEMPLATES, k=1) if has_minor_con else []
        
        sentences = [p[0] for p in pros_pick]
        if cons_pick:
            sentences.append("On the downside, " + cons_pick[0][0].lower())
        sentences.append(random.choice(FILLERS_POS))
        
        title = pros_pick[0][1]
        text = " ".join(sentences)
        
    elif rand_val < 0.73:
        # Neutral review
        rating = 3
        neu_pick = random.choice(NEUTRAL_TEMPLATES)
        pro_pick = random.choice(PROS_TEMPLATES)
        con_pick = random.choice(CONS_TEMPLATES)
        
        sentences = [neu_pick[0], pro_pick[0], "However, " + con_pick[0].lower()]
        title = neu_pick[1]
        text = " ".join(sentences)
        
    else:
        # Negative review
        rating = random.choice([1, 2])
        cons_pick = random.sample(CONS_TEMPLATES, k=random.randint(2, 3))
        has_minor_pro = random.random() < 0.3
        pros_pick = random.sample(PROS_TEMPLATES, k=1) if has_minor_pro else []
        
        sentences = []
        if pros_pick:
            sentences.append("While " + pros_pick[0][0].lower() + ",")
        sentences.extend([c[0] for c in cons_pick])
        sentences.append(random.choice(FILLERS_NEG))
        
        title = cons_pick[0][1]
        text = " ".join(sentences)

    base_date = datetime(2025, 1, 1)
    review_date = (base_date + timedelta(days=random.randint(0, 360))).strftime("%Y-%m-%d")
    verified = random.random() < 0.88
    helpful = random.choices([0, 1, 2, 5, 12, 34, 89], weights=[40, 25, 15, 10, 5, 3, 2])[0]
    
    return {
        "review_id": f"REV-{review_idx:04d}",
        "product_name": PRODUCT_NAME,
        "rating": rating,
        "review_title": title,
        "review_text": text,
        "verified_purchase": verified,
        "review_date": review_date,
        "helpful_votes": helpful
    }

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sample_1000_reviews.csv")
    
    reviews = [generate_review(i + 1) for i in range(1000)]
    
    fieldnames = [
        "review_id", "product_name", "rating", "review_title",
        "review_text", "verified_purchase", "review_date", "helpful_votes"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)
        
    print(f"Successfully generated {len(reviews)} reviews into {output_file}")

if __name__ == "__main__":
    main()
