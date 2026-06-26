import { describe, it, expect } from "vitest";
import {
  nextPass,
  initialState,
  isDone,
  draftSizeForMode,
  previewAcceptedCount,
  type StagingToken,
} from "./mtp";

const PHRASE = ["The", "capital", "of", "France", "is", "Paris", ",", "which", "is", "beautiful", "."];

function correctStaging(confirmedLen: number, count: number, idStart = 0): StagingToken[] {
  return Array.from({ length: count }, (_, i) => ({
    id: idStart + i,
    text: PHRASE[confirmedLen + i] ?? "…",
  }));
}

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
    expect(s.lastVerification).toEqual([]);
  });
});

describe("isDone", () => {
  it("false before any passes", () => expect(isDone(initialState(), PHRASE)).toBe(false));
  it("false when confirmed full but staging not empty", () => {
    const s = { ...initialState(), confirmed: [...PHRASE], staging: [{ id: 0, text: "extra" }], passes: 1 };
    expect(isDone(s, PHRASE)).toBe(false);
  });
  it("true when confirmed full, no staging, passes > 0", () => {
    const s = { ...initialState(), confirmed: [...PHRASE], passes: 5 };
    expect(isDone(s, PHRASE)).toBe(true);
  });
});

describe("previewAcceptedCount", () => {
  it("returns 0 when staging is empty", () => {
    expect(previewAcceptedCount(initialState(), PHRASE)).toBe(0);
  });
  it("returns correct count when all staging tokens match", () => {
    const s = {
      ...initialState(),
      confirmed: ["The"],
      staging: [{ id: 0, text: "capital" }, { id: 1, text: "of" }],
    };
    expect(previewAcceptedCount(s, PHRASE)).toBe(2);
  });
  it("stops at first mismatch", () => {
    const s = {
      ...initialState(),
      confirmed: ["The"],
      staging: [{ id: 0, text: "capital" }, { id: 1, text: "WRONG" }, { id: 2, text: "France" }],
    };
    expect(previewAcceptedCount(s, PHRASE)).toBe(1);
  });
  it("returns 0 when first token is wrong", () => {
    const s = { ...initialState(), confirmed: [], staging: [{ id: 0, text: "wrong" }] };
    expect(previewAcceptedCount(s, PHRASE)).toBe(0);
  });
});

describe("nextPass — standard mode", () => {
  it("emits exactly 1 token per pass", () => {
    const s = nextPass(initialState(), "standard", PHRASE, []);
    expect(s.confirmed).toEqual(["The"]);
    expect(s.staging).toEqual([]);
    expect(s.passes).toBe(1);
    expect(s.lastVerification).toEqual([]);
  });

  it("accumulates all tokens over enough passes", () => {
    let s = initialState();
    for (let i = 0; i < PHRASE.length; i++) s = nextPass(s, "standard", PHRASE, []);
    expect(s.confirmed).toEqual(PHRASE);
  });

  it("does not overshoot the phrase", () => {
    let s = initialState();
    for (let i = 0; i < PHRASE.length + 5; i++) s = nextPass(s, "standard", PHRASE, []);
    expect(s.confirmed.length).toBe(PHRASE.length);
  });
});

describe("nextPass — mtp1 mode, all correct", () => {
  it("first pass: 1 confirmed + provided staging", () => {
    const staging = correctStaging(1, 1);
    const s = nextPass(initialState(), "mtp1", PHRASE, staging);
    expect(s.confirmed).toEqual(["The"]);
    expect(s.staging).toEqual(staging);
    expect(s.lastVerification).toEqual([]);
    expect(s.passes).toBe(1);
  });

  it("second pass: staging token accepted, main head advances, new staging adopted", () => {
    let s = nextPass(initialState(), "mtp1", PHRASE, correctStaging(1, 1));
    const newStaging = correctStaging(3, 1, 10);
    s = nextPass(s, "mtp1", PHRASE, newStaging);
    expect(s.confirmed).toEqual(["The", "capital", "of"]);
    expect(s.staging).toEqual(newStaging);
    expect(s.lastVerification[0].status).toBe("accepted");
  });

  it("completes in fewer passes than standard", () => {
    let sMtp = initialState();
    let sSt = initialState();
    while (!isDone(sMtp, PHRASE)) {
      const nextLen = sMtp.confirmed.length + sMtp.staging.filter((_, i) => {
        return sMtp.staging[i].text === PHRASE[sMtp.confirmed.length + i];
      }).length + 1;
      sMtp = nextPass(sMtp, "mtp1", PHRASE, correctStaging(nextLen, 1, sMtp.idCounter));
    }
    while (!isDone(sSt, PHRASE)) sSt = nextPass(sSt, "standard", PHRASE, []);
    expect(sMtp.passes).toBeLessThan(sSt.passes);
  });
});

describe("nextPass — verification with rejections", () => {
  it("marks first mismatch as rejected, rest as discarded", () => {
    const wrongStaging: StagingToken[] = [
      { id: 0, text: "capital" },  // correct
      { id: 1, text: "WRONG" },    // wrong
      { id: 2, text: "France" },   // discarded (after rejection)
    ];
    const s = nextPass(
      { ...initialState(), staging: wrongStaging },
      "mtp3",
      PHRASE,
      []
    );
    expect(s.lastVerification[0].status).toBe("accepted");
    expect(s.lastVerification[1].status).toBe("rejected");
    expect(s.lastVerification[1].predicted).toBe("WRONG");
    expect(s.lastVerification[1].truth).toBe("of");
    expect(s.lastVerification[2].status).toBe("discarded");
  });

  it("rejection stops acceptance — confirmed only gets tokens before rejection + 1 from main head", () => {
    const wrongStaging: StagingToken[] = [
      { id: 0, text: "WRONG" },
      { id: 1, text: "of" },
    ];
    const s = nextPass(
      { ...initialState(), staging: wrongStaging },
      "mtp1",
      PHRASE,
      []
    );
    // 0 accepted from staging, main head emits "The"
    expect(s.confirmed).toEqual(["The"]);
    expect(s.lastVerification[0].status).toBe("rejected");
  });

  it("all rejected staging still advances by 1 via main head", () => {
    const wrongStaging: StagingToken[] = [{ id: 0, text: "NOPE" }, { id: 1, text: "NOPE2" }, { id: 2, text: "NOPE3" }];
    const s = nextPass({ ...initialState(), staging: wrongStaging }, "mtp3", PHRASE, []);
    expect(s.confirmed.length).toBe(1);
    expect(s.confirmed[0]).toBe(PHRASE[0]);
  });

  it("all accepted staging advances by staging.length + 1", () => {
    const staging: StagingToken[] = [
      { id: 0, text: "capital" },
      { id: 1, text: "of" },
      { id: 2, text: "France" },
    ];
    const s = nextPass({ ...initialState(), staging }, "mtp3", PHRASE, []);
    // accepted capital, of, France → main head emits "is"
    expect(s.confirmed).toEqual(["The", "capital", "of", "France", "is"]);
    expect(s.lastVerification.every((v) => v.status === "accepted")).toBe(true);
  });

  it("first-token rejection: all staging marked rejected or discarded", () => {
    const staging: StagingToken[] = [
      { id: 0, text: "WRONG" },
      { id: 1, text: "capital" },
      { id: 2, text: "of" },
    ];
    const s = nextPass({ ...initialState(), staging }, "mtp3", PHRASE, []);
    expect(s.lastVerification[0].status).toBe("rejected");
    expect(s.lastVerification[1].status).toBe("discarded");
    expect(s.lastVerification[2].status).toBe("discarded");
  });
});

describe("nextPass — idCounter", () => {
  it("adopts newStagingTokens idCounter from the provided array", () => {
    const newStaging = [{ id: 99, text: "capital" }];
    const s = nextPass(initialState(), "mtp1", PHRASE, newStaging);
    expect(s.staging[0].id).toBe(99);
  });
});
