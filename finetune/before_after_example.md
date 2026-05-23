# Fine-tune before/after

Same prompt, greedy decoding, run through the base model and then the base plus
the LoRA adapter trained on databricks-dolly-15k (200 steps, bf16, r=16). The
adapter shifts the model from the base Instruct model's long multi-section
default toward dolly's more concise, single-paragraph human style. This is the
honest evidence that the fine-tune stage of the loop actually moved the model.

Training: loss fell from 1.98 to 1.21 over 200 steps. Adapter is 0.46 percent of
the 14B parameters (68.8M trainable).

**Prompt:** Explain why the sky is blue to a ten year old.

## Base (Qwen2.5-14B-Instruct)

Sure! Imagine you're looking up at the sky on a sunny day. The sky looks blue because of something called "Rayleigh scattering." Let's break that down into simpler ideas:

1. Sunlight and Colors: Sunlight is made up of all colors of light, like in a rainbow. But we can't see all these colors separately when we look at sunlight.
2. Air and Light: When sunlight reaches Earth, it has to pass through our atmosphere, which is full of tiny particles and molecules of air.
3. Scattering: As the sunlight travels through the air, the light bounces off these tiny particles. Different colors of light scatter differently. Blue light scatters more than other colors because it travels as shorter, smaller waves.
4. Seeing Blue: Because blue light scatters more, it spreads out in all directions much more than other colors. So, when you look up at the sky, your eyes catch mostly blue light coming from all over the [...]

## Base + LoRA (dolly-tuned)

The sky is blue because of something called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it collides with molecules in the air and gets scattered in all directions. Blue light has shorter wavelengths than other colors, so it scatters more easily and is more likely to reach our eyes. This is why we see the sky as blue during the day.
