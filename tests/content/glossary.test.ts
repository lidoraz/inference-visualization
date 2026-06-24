import { describe, it, expect } from "vitest";
import { glossary } from "../../src/content/glossary";
import { stageGuides } from "../../src/content/stageGuides";

describe("glossary", () => {
  it("contains all required keys", () => {
    const required = [
      "prefill",
      "decode",
      "kvCache",
      "block",
      "blockTable",
      "preemption",
      "swapping",
      "continuousBatching",
      "tokenBudget",
      "maxBatchSize",
      "vocabSampling",
      "tokenizer",
      "pagedAttention",
      "internalFragmentation",
      "waitingQueue",
      "runningQueue",
    ];
    for (const key of required) {
      expect(glossary, `glossary is missing required key: "${key}"`).toHaveProperty(key);
    }
  });

  it("every entry has non-empty term, short, and long fields", () => {
    for (const [key, entry] of Object.entries(glossary)) {
      expect(entry.term.length, `glossary["${key}"].term is empty`).toBeGreaterThan(0);
      expect(entry.short.length, `glossary["${key}"].short is empty`).toBeGreaterThan(0);
      expect(entry.long.length, `glossary["${key}"].long is empty`).toBeGreaterThan(0);
    }
  });

  it("short text is within the ~90-char guideline", () => {
    for (const [key, entry] of Object.entries(glossary)) {
      expect(
        entry.short.length,
        `glossary["${key}"].short exceeds 90 chars (${entry.short.length})`
      ).toBeLessThanOrEqual(90);
    }
  });
});

describe("stageGuides cross-references", () => {
  it("every glossaryKey referenced in stageGuides exists in glossary", () => {
    for (const [stageId, guide] of Object.entries(stageGuides)) {
      for (const step of guide.steps) {
        // Overview/aside steps may omit glossaryKey (they use `title` instead).
        if (!step.glossaryKey) continue;
        expect(
          glossary,
          `stageGuides[${stageId}] references unknown glossaryKey: "${step.glossaryKey}"`
        ).toHaveProperty(step.glossaryKey);
      }
    }
  });

  it("inline [[term]] markup in guide notes references known glossary keys", () => {
    const re = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g;
    for (const [stageId, guide] of Object.entries(stageGuides)) {
      for (const step of guide.steps) {
        if (!step.note) continue;
        let m: RegExpExecArray | null;
        while ((m = re.exec(step.note)) !== null) {
          const key = m[1].trim();
          expect(
            glossary,
            `stageGuides[${stageId}] note references unknown [[${key}]]`
          ).toHaveProperty(key);
        }
      }
    }
  });

  it("stages 1, 2, and 3 are all defined", () => {
    expect(stageGuides).toHaveProperty("1");
    expect(stageGuides).toHaveProperty("2");
    expect(stageGuides).toHaveProperty("3");
  });

  it("each stage has a non-empty title and at least one step", () => {
    for (const [stageId, guide] of Object.entries(stageGuides)) {
      expect(guide.title.length, `stageGuides[${stageId}].title is empty`).toBeGreaterThan(0);
      expect(guide.steps.length, `stageGuides[${stageId}].steps is empty`).toBeGreaterThan(0);
    }
  });

  it("stageId field matches the record key", () => {
    for (const [key, guide] of Object.entries(stageGuides)) {
      expect(guide.stageId, `stageGuides[${key}].stageId does not match key`).toBe(Number(key));
    }
  });
});
