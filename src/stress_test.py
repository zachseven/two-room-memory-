"""
Stress Test: 100 adversarial/edge-case examples
Designed to probe the boundary of the triviality gate
"""

# Test cases: (exchange, expected)
# "flush" = trivial, "persist" = meaningful

STRESS_TEST_CASES = [
    # === 50 TRIVIAL (should flush) ===
    ("What's the difference between a crocodile and an alligator?", "flush"),
    ("How many teaspoons in a tablespoon?", "flush"),
    ("The sunset was pretty tonight", "flush"),
    ("I wonder if it'll snow this weekend", "flush"),
    ("What's the best way to reheat pizza?", "flush"),
    ("Dogs are better than cats honestly", "flush"),
    ("I can never remember which way to turn a screw", "flush"),
    ("What does per se mean?", "flush"),
    ("The wifi here is terrible", "flush"),
    ("Is a hot dog a sandwich?", "flush"),
    ("I think I left the oven on", "flush"),
    ("What's the population of Canada?", "flush"),
    ("This coffee is too bitter", "flush"),
    ("Why do they call it a building if it's already built?", "flush"),
    ("I prefer window seats on planes", "flush"),
    ("What year did the iPhone come out?", "flush"),
    ("The traffic was insane this morning", "flush"),
    ("Do you know any good podcasts?", "flush"),
    ("I always forget how to fold a fitted sheet", "flush"),
    ("What's the shortest day of the year?", "flush"),
    ("My neighbor's dog won't stop barking", "flush"),
    ("Is mercury really in retrograde?", "flush"),
    ("I should probably drink more water", "flush"),
    ("What's a good substitute for butter in baking?", "flush"),
    ("The new season of that show is mid", "flush"),
    ("Why do we park in driveways and drive on parkways?", "flush"),
    ("I can't find my keys anywhere", "flush"),
    ("What's the capital of Australia?", "flush"),
    ("The moon looks huge tonight", "flush"),
    ("I need to get my oil changed", "flush"),
    ("Is it fewer or less?", "flush"),
    ("These chips are stale", "flush"),
    ("What time zone is Arizona in?", "flush"),
    ("I hate when my phone dies", "flush"),
    ("The grocery store was packed today", "flush"),
    ("What's a good stretch for lower back?", "flush"),
    ("I think pineapple on pizza is fine actually", "flush"),
    ("Why do they put braille on drive-through ATMs?", "flush"),
    ("I can never remember my password", "flush"),
    ("What's the deal with daylight saving time?", "flush"),
    ("My plant is dying I think", "flush"),
    ("Is it effect or affect?", "flush"),
    ("The new update broke everything", "flush"),
    ("What's the best airline?", "flush"),
    ("I forgot to take my vitamins", "flush"),
    ("Why do we say heads up when we mean duck?", "flush"),
    ("This pen doesn't work", "flush"),
    ("What's a normal body temperature?", "flush"),
    ("I should really clean my car", "flush"),
    ("How long do hard boiled eggs last?", "flush"),
    
    # === 50 MEANINGFUL (should persist) ===
    ("My dad and I haven't spoken in three years", "persist"),
    ("I think I'm burning out at work", "persist"),
    ("I've been having the same nightmare for weeks", "persist"),
    ("My kid got diagnosed with dyslexia", "persist"),
    ("I'm considering leaving my religion", "persist"),
    ("I've never told anyone this but I was assaulted", "persist"),
    ("My spouse and I are in couples therapy", "persist"),
    ("I think I might be on the spectrum", "persist"),
    ("I grew up as a foster kid", "persist"),
    ("I've been self-harming again", "persist"),
    ("My mom is showing signs of dementia", "persist"),
    ("I'm the only one in my family who went to college", "persist"),
    ("I don't feel anything anymore", "persist"),
    ("My business partner screwed me over", "persist"),
    ("I was a caregiver for my brother until he died", "persist"),
    ("I have a hard time forming attachments", "persist"),
    ("I'm finally coming out to my family this week", "persist"),
    ("I lost custody of my kids", "persist"),
    ("I've been sober for 90 days", "persist"),
    ("I don't know if I want to be alive sometimes", "persist"),
    ("My identity was stolen and it ruined my credit", "persist"),
    ("I was homeschooled by abusive parents", "persist"),
    ("I just found out I'm adopted", "persist"),
    ("I can't afford my medication anymore", "persist"),
    ("My wife had a miscarriage last month", "persist"),
    ("I'm undocumented and scared every day", "persist"),
    ("I grew up without running water", "persist"),
    ("I've been the target of a stalker", "persist"),
    ("I took care of my siblings while my parents worked", "persist"),
    ("I have chronic pain that doctors can't explain", "persist"),
    ("I'm neurodivergent and masking exhausts me", "persist"),
    ("My family lost everything in a fire", "persist"),
    ("I've never had a healthy relationship", "persist"),
    ("I'm estranged from all my siblings", "persist"),
    ("I age out of foster care next month", "persist"),
    ("I witnessed domestic violence as a child", "persist"),
    ("I've been rejected from every job I applied to", "persist"),
    ("My therapist says I have attachment trauma", "persist"),
    ("I'm the executor of my mom's estate and it's destroying me", "persist"),
    ("I can't connect with people my own age", "persist"),
    ("I was bullied so bad I switched schools three times", "persist"),
    ("I have an autoimmune disease that limits what I can do", "persist"),
    ("I lost my best friend to suicide", "persist"),
    ("My parent went to prison when I was young", "persist"),
    ("I've been in and out of psych holds", "persist"),
    ("I dropped out to take care of my sick parent", "persist"),
    ("I don't feel safe in my own home", "persist"),
    ("I'm grieving someone who's still alive", "persist"),
    ("I've spent my whole life being the responsible one", "persist"),
    ("I don't know who I am outside of work", "persist"),
]


def run_stress_test():
    """Run the stress test against the classifier"""
    from classifier_gate import predict
    
    correct = 0
    false_positives = []  # Should persist, got flushed (BAD)
    false_negatives = []  # Should flush, got persisted (less bad)
    
    print("=" * 70)
    print("STRESS TEST: 100 Adversarial Examples")
    print("=" * 70)
    
    for exchange, expected in STRESS_TEST_CASES:
        prediction, confidence = predict(exchange)
        predicted = prediction.lower()
        
        if predicted == expected:
            correct += 1
        else:
            if expected == "persist" and predicted == "flush":
                false_positives.append((exchange, confidence))
            else:
                false_negatives.append((exchange, confidence))
    
    accuracy = correct / len(STRESS_TEST_CASES)
    
    print(f"\nAccuracy: {correct}/{len(STRESS_TEST_CASES)} = {accuracy:.1%}")
    print(f"False Positives (lost memories): {len(false_positives)}")
    print(f"False Negatives (noise persisted): {len(false_negatives)}")
    
    if false_positives:
        print(f"\n{'=' * 70}")
        print("FALSE POSITIVES — Should PERSIST, got flushed (CRITICAL)")
        print("=" * 70)
        for ex, conf in false_positives:
            print(f"  [{conf:.2f}] {ex}")
    
    if false_negatives:
        print(f"\n{'=' * 70}")
        print("FALSE NEGATIVES — Should FLUSH, got persisted (less critical)")
        print("=" * 70)
        for ex, conf in false_negatives:
            print(f"  [{conf:.2f}] {ex}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(STRESS_TEST_CASES)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Critical errors (lost memories): {len(false_positives)}")
    print(f"Minor errors (noise kept): {len(false_negatives)}")
    
    if accuracy >= 0.95:
        print("\n✓ EXCELLENT — Gate holds up under stress")
    elif accuracy >= 0.90:
        print("\n~ GOOD — Gate is robust but has edge cases")
    elif accuracy >= 0.80:
        print("\n! FAIR — Gate needs more training data")
    else:
        print("\n✗ POOR — Gate needs significant work")
    
    return {
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


if __name__ == "__main__":
    run_stress_test()
