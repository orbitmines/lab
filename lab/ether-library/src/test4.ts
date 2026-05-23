/**
 * test4.ts — Programmatic access to the x86 instruction set, sourced from
 * XED's own compiled-in tables via the `xed-tables` example binary.
 *
 * We don't parse XED datafiles ourselves. `xed-tables` walks the in-memory
 * instruction table that XED's build emits from those datafiles, so the
 * fields below are exactly what XED's C library reports.
 *
 *   - loadInstructions(bin?) → Instruction[]
 *   - each Instruction carries iclass / iform / category / extension / isaSet /
 *     attributes, plus an operand list with name / visibility / action / type /
 *     xtype / details, every value verbatim from xed-tables.
 *
 * Binary path defaults to $XED_TABLES_BIN, then the wkit build at
 * ~/Documents/github.com/intelxed/xed/obj/wkit/bin/xed-tables.
 */

import { spawnSync } from "node:child_process";
import * as path     from "node:path";

const DEFAULT_BIN = process.env.XED_TABLES_BIN ?? path.join(
  process.env.HOME ?? "",
  "Documents/github.com/intelxed/xed/obj/wkit/bin/xed-tables"
);

export interface Operand {
  index:      number;
  /** REG0, MEM0, IMM0, RELBR, BASE0, INDEX, SCALE, … */
  name:       string;
  /** EXPLICIT / IMPLICIT / SUPPRESSED */
  visibility: string;
  /** R / W / RW / CR / CW / CRW */
  action:     string;
  /** REG / IMM_CONST / NT_LOOKUP_FN / … */
  type:       string;
  /** F32 / F64 / F80 / I8 / U32 / INVALID / … */
  xtype:      string;
  /** Operand-source detail: ST(0), AL, GPR64_R, X87, etc. Optional. */
  details?:   string;
}

export interface Instruction {
  index:      number;
  iclass:     string;
  iform:      string;
  category:   string;
  extension:  string;
  isaSet:     string;
  attributes: string[];
  operands:   Operand[];
}

export function loadInstructions(bin: string = DEFAULT_BIN): Instruction[] {
  const r = spawnSync(bin, [], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  if (r.error)       throw new Error(`Failed to run ${bin}: ${r.error.message}`);
  if (r.status !== 0) throw new Error(`${bin} exited ${r.status}: ${r.stderr}`);

  const out: Instruction[] = [];
  let cur: Instruction | null = null;

  for (const rawLine of r.stdout.split("\n")) {
    if (!rawLine) continue;

    if (rawLine.startsWith("\t")) {
      // Operand line: "\t<idx> <name> <vis> <action> <type> <xtype> [<details>...]"
      if (!cur) continue;
      const toks = rawLine.trim().split(/\s+/);
      if (toks.length < 6) continue;
      cur.operands.push({
        index:      parseInt(toks[0], 10),
        name:       toks[1],
        visibility: toks[2],
        action:     toks[3],
        type:       toks[4],
        xtype:      toks[5],
        details:    toks.length > 6 ? toks.slice(6).join(" ") : undefined,
      });
      continue;
    }

    // Header line: "<idx> <ICLASS> <IFORM> <CATEGORY> <EXTENSION> <ISA_SET> ATTRIBUTES: [attrs...]"
    const headerMatch = rawLine.match(/^(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+ATTRIBUTES:\s*(.*)$/);
    if (headerMatch) {
      if (cur) out.push(cur);
      cur = {
        index:      parseInt(headerMatch[1], 10),
        iclass:     headerMatch[2],
        iform:      headerMatch[3],
        category:   headerMatch[4],
        extension:  headerMatch[5],
        isaSet:     headerMatch[6],
        attributes: headerMatch[7].trim() ? headerMatch[7].trim().split(/\s+/) : [],
        operands:   [],
      };
      continue;
    }

    // Operand-count line (" 3 ") and any other lines are ignored.
  }
  if (cur) out.push(cur);
  return out;
}

// ── CLI ─────────────────────────────────────────────────────────────────────
if (process.argv[1] && import.meta.url === `file://${path.resolve(process.argv[1])}`) {
  const bin    = process.argv[2] || DEFAULT_BIN;
  const instrs = loadInstructions(bin);
  console.log(JSON.stringify(instrs, null, 2))

  const iclasses   = new Set(instrs.map(i => i.iclass));
  const extensions = new Set(instrs.map(i => i.extension).filter(Boolean));
  const arityHist  = new Map<number, number>();
  for (const i of instrs) arityHist.set(i.operands.length, (arityHist.get(i.operands.length) ?? 0) + 1);

  console.log(`source:      ${bin}`);
  console.log(`forms:       ${instrs.length}`);
  console.log(`iclasses:    ${iclasses.size}`);
  console.log(`extensions:  ${extensions.size}`);
  console.log(`arity histogram (# operands → # forms):`);
  for (const k of [...arityHist.keys()].sort((a, b) => a - b)) {
    console.log(`  ${k}: ${arityHist.get(k)}`);
  }

  const filter = process.argv[3];
  if (filter) {
    console.log(`\nforms for ICLASS=${filter}:`);
    for (const i of instrs.filter(x => x.iclass === filter)) {
      console.log(`\n  ${i.iform}  [${i.extension}/${i.isaSet}]  attrs=[${i.attributes.join(" ")}]`);
      for (const op of i.operands) {
        const d = op.details ? ` (${op.details})` : "";
        console.log(`    ${op.index}: ${op.name}  ${op.visibility} ${op.action}  ${op.type}/${op.xtype}${d}`);
      }
    }
  }
}
