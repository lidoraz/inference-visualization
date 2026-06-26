/**
 * Sample prompts for the "Random" add-request button. UI-only data — picking is
 * allowed to use Math.random here (the simulation engine stays deterministic;
 * this just seeds user input, same as typing).
 */

export const sampleSentences: string[] = [
  "The quick brown fox jumps over the lazy dog",
  "Explain how paged attention reduces memory fragmentation",
  "Translate this sentence into French please",
  "Once upon a time in a distant galaxy",
  "Summarize the key ideas behind continuous batching",
  "What is the capital of the smallest country in Europe",
  "Write a haiku about garbage collection in Rust",
  "Large language models serve many users at once",
  "Describe the difference between prefill and decode phases",
  "Roses are red violets are blue inference is fast",
  "The capital of France is Paris which is beautiful",
];

/** Pick a random sample sentence (UI-only; non-deterministic by design). */
export function randomSentence(): string {
  const i = Math.floor(Math.random() * sampleSentences.length);
  return sampleSentences[i];
}
