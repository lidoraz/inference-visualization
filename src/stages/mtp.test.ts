import { describe, it, expect } from "vitest";
import {
  nextPass,
  initialState,
  isDone,
  draftSizeForMode,
} from "./mtp";

const PHRASE = ["The", "capital", "of", "France", "is", "Paris", ",", "which", "is", "beautiful", "."];

describe("draftSizeForMode", () => {
  it("standard → 0", () => expect(draftSizeForMode("standard")).toBe(0));
  it("mtp1 → 1", () => expect(draftSizeForMode("mtp1")).toBe(1));
  it("mtp3 → 3", () => expect(draftSizeForMode("mtp3")).toBe(3));
});

describe("initialState", () => {
  it("starts empty", () => {
    const s = initialState();
    expect(s.confirmed).toEqual([]);
    expect(s.staging).toEqual([]);
    expect(s.passes).toBe(0);
    expect(s.idCounter).toBe(0);
  });
});

describe("isDone", () => {
  it("false before any passes", () => {
    expect(isDone(initialState(), PHRASE)).toBe(false);
  });

  it("false when confirmed is full but staging still has tokens", () => {
    const s = { confirmed: [...PHRASE], staging: [{ id: 0, text: "extra" }], passes: 1, idCounter: 1 };
    expect(isDone(s, PHRASE)).toBe(false);
  });

  it("true when confirmed equals phrase length, no staging, passes > 0", () => {
    const s = { confirmed: [...PHRASE], staging: [], passes: 5, idCounter: 0 };
    expect(isDone(s, PHRASE)).toBe(true);
  });
});

describe("nextPass — standard mode", () => {
  it("emits exactly 1 token per pass", () => {
    let s = initialState();
    s = nextPass(s, "standard", PHRASE);
    expect(s.confirmed).toEqual(["The"]);
    expect(s.staging).toEqual([]);
    expect(s.passes).toBe(1);
  });

  it("accumulates tokens over multiple passes", () => {
    let s = initialState();
    for (let i = 0; i < PHRASE.length; i++) {
      s = nextPass(s, "standard", PHRASE);
    }
    expect(s.confirmed).toEqual(PHRASE);
    expect(s.passes).toBe(PHRASE.length);
  });

  it("does not overshoot the phrase", () => {
    let s = initialState();
    for (let i = 0; i < PHRASE.length + 5; i++) {
      s = nextPass(s, "standard", PHRASE);
    }
    expect(s.confirmed.length).toBe(PHRASE.length);
  });
});

describe("nextPass — mtp1 mode", () => {
  it("first pass: 1 confirmed + 1 staging", () => {
    const s = nextPass(initialState(), "mtp1", PHRASE);
    expect(s.confirmed).toEqual(["The"]);
    expect(s.staging.length).toBe(1);
    expect(s.staging[0].text).toBe("capital");
    expect(s.passes).toBe(1);
  });

  it("second pass: staging token is accepted, main head emits next, new staging predicted", () => {
    let s = nextPass(initialState(), "mtp1", PHRASE);
    s = nextPass(s, "mtp1", PHRASE);
    // staging "capital" was accepted, main head emitted "of", new staging is "France"
    expect(s.confirmed).toEqual(["The", "capital", "of"]);
    expect(s.staging.length).toBe(1);
    expect(s.staging[0].text).toBe("France");
  });

  it("completes the full phrase in fewer passes than standard", () => {
    let s = initialState();
    let standardPasses = 0;
    let sStd = initialState();
    while (!isDone(s, PHRASE)) s = nextPass(s, "mtp1", PHRASE);
    while (!isDone(sStd, PHRASE)) { sStd = nextPass(sStd, "standard", PHRASE); standardPasses++; }
    expect(s.passes).toBeLessThan(standardPasses);
    expect(s.confirmed).toEqual(PHRASE);
  });

  it("no staging after the phrase is exhausted", () => {
    let s = initialState();
    while (!isDone(s, PHRASE)) s = nextPass(s, "mtp1", PHRASE);
    expect(s.staging).toEqual([]);
  });
});

describe("nextPass — mtp3 mode", () => {
  it("first pass: 1 confirmed + 3 staging", () => {
    const s = nextPass(initialState(), "mtp3", PHRASE);
    expect(s.confirmed).toEqual(["The"]);
    expect(s.staging.length).toBe(3);
    expect(s.staging.map((t) => t.text)).toEqual(["capital", "of", "France"]);
  });

  it("second pass: all 3 staging accepted + 1 main head + 3 new staging", () => {
    let s = nextPass(initialState(), "mtp3", PHRASE);
    s = nextPass(s, "mtp3", PHRASE);
    // accepted: capital, of, France; main head: is; staging: Paris, ,, which
    expect(s.confirmed).toEqual(["The", "capital", "of", "France", "is"]);
    expect(s.staging.map((t) => t.text)).toEqual(["Paris", ",", "which"]);
  });

  it("completes in fewer passes than mtp1", () => {
    let s1 = initialState(), s3 = initialState();
    while (!isDone(s1, PHRASE)) s1 = nextPass(s1, "mtp1", PHRASE);
    while (!isDone(s3, PHRASE)) s3 = nextPass(s3, "mtp3", PHRASE);
    expect(s3.passes).toBeLessThan(s1.passes);
  });

  it("staging shrinks near end of phrase without overshooting", () => {
    let s = initialState();
    while (!isDone(s, PHRASE)) s = nextPass(s, "mtp3", PHRASE);
    expect(s.confirmed.length).toBe(PHRASE.length);
    expect(s.staging).toEqual([]);
  });

  it("staging ids are unique across passes", () => {
    let s = initialState();
    const allIds: number[] = [];
    while (!isDone(s, PHRASE)) {
      s = nextPass(s, "mtp3", PHRASE);
      allIds.push(...s.staging.map((t) => t.id));
    }
    expect(new Set(allIds).size).toBe(allIds.length);
  });
});

describe("mode transitions preserve determinism", () => {
  it("switching from standard to mtp1 on a short phrase always yields the same result", () => {
    const SHORT = ["a", "b", "c", "d"];
    let s = nextPass(initialState(), "standard", SHORT); // confirmed: ["a"], staging: []
    s = nextPass(s, "mtp1", SHORT);                      // confirms "b", stages "c"
    expect(s.confirmed).toEqual(["a", "b"]);
    expect(s.staging.map((t) => t.text)).toEqual(["c"]);
  });
});
