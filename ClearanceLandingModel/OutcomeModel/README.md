# Headed Clearance Outcome Model

This folder adds a decision-quality layer to the headed-clearance landing model.

## What is modeled

- `p_relief_success`: probability the clearance produces relief, based on the situation before the header.
- `p_opponent_shot_10s`: probability the opponent produces a shot within 10 seconds.
- `relief_oe`: actual relief outcome minus expected relief probability.
- `shot_prevention_oe`: expected shot risk minus actual shot outcome. Positive means the player avoided a shot that the model considered plausible.
- `decision_quality_oe`: composite decision score combining the existing landing-value score with relief over expected and shot prevention over expected.

## Important interpretation

These are analysis scores, not scouting absolutes. Use them to find clips and patterns:

- who turns bad situations into relief,
- who clears to areas with low second-wave risk,
- who gets same-team first touch more than expected,
- which clearance styles lead to danger.
