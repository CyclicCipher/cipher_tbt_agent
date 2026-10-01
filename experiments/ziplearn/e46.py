"""E46 -- a Brainfuck interpreter as a WRITTEN looped block (ziplearn DESIGN §18.3). No training, no fitting.

The program space of Self-Play Pretraining with Zero Data (arXiv 2609.30063, Appendix E) is a Brainfuck machine:
eight instructions `< > + - [ ] . ,` and an end token `F`, a circular tape, cells modulo 256, `,` reading i.i.d.
uniform random bytes, unmatched brackets treated as no-ops, and ten single-character macros DEFINED as expansions
into pure Brainfuck (their Table 4). This file writes that machine into the weights of two `h1_lid.Block`s and runs
it as a looped transformer: ONE PASS = ONE BRAINFUCK STEP, the program and the random input as tokens in the
context, the weights identical for every program.

Tokens (one sequence): L program tokens (position, opcode, precomputed bracket target), N tape tokens (position,
byte), K input tokens (position, random byte), one REGISTER (instruction pointer `ip`, data pointer `dp`, input
pointer `inp`), one NULL sink. A byte is two one-hot nibbles, so +1 mod 256 is a nibble shift plus a carry that is a
conjunction of one-hot dims -- one threshold row, no extra depth.

A pass, in the substrate's own modules:
  block A  attention  fetch   register <- prog[ip]      opcode, bracket target        `Match` (position) + null sink
                      read    register <- tape[dp]      the current cell              `Match`
                      inread  register <- input[inp]    the next random byte          `Match`
           MLP        the new cell value (+ - , or unchanged), zero test, output, halt  `Row`s (threshold units)
  block B  attention  write   tape[dp] <- register      the new value, gated "here"   `Match` from the tape side,
                                                                                     null at half strength = the gate
           MLP        next ip (incl. the conditional jumps of [ and ]), next dp, next inp; tape cell <- new value
  BoundaryOp          Quantise ip/dp/inp/bytes to one-hot, Keep/Clear the scratch, Halt on F   (h1_lid.BoundaryOp)
The runner reads exactly three things off the register -- `halt`, `emit`, `out` (a `Readout`) -- and nothing else.

Verification is against `run_reference`, a plain Python interpreter with the same semantics, byte for byte on the
output AND on the final tape and data pointer. The sweep uses a sparse kernel on the SAME weight tensors for speed;
`check_dense` runs the literal `h1_lid.Block.forward` on a subset and asserts the two agree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transformers"))
import h1_lid as H1                                                    # noqa: E402

# ------------------------------------------------------------------------------------------------- the machine's language
BASE = "><+-[].,"
MACROS = {                                                             # 2609.30063 Table 4, verbatim expansions
    "Z": "[-]", "R": "[->+<]", "L": "[->+++<]", "N": "[-<->]", "C": "[->+>+<<]",
    "G": "[>]", "H": "[<]", "W": "[[-]>+<]", "V": "[.>]", "X": "[-]" + "+" * 16,
}
ALPHABET = BASE + "F" + "".join(MACROS)                                # the 19-symbol generator alphabet
OPS = [">", "<", "+", "-", "[", "]", ".", ",", "F", "NOP"]             # the machine's opcodes after expansion
OP = {o: i for i, o in enumerate(OPS)}


def expand(prog: str) -> str:
    """Macros expanded as Table 4 defines them; anything after expansion is pure Brainfuck plus F."""
    return "".join(MACROS.get(c, c) for c in prog)


def compile_program(prog: str, L: int):
    """(opcodes, bracket targets) for an L-slot program memory. Brackets are matched with a stack on the expanded
    string; an unmatched bracket becomes NOP (Appendix E: "unmatched brackets, which we treat as no-ops"). Targets
    point PAST the partner: `[` at i with partner j jumps to j+1, `]` at j jumps to i+1. One F is appended and the
    rest padded with F, so the instruction pointer always lands on F."""
    s = expand(prog)
    if len(s) + 1 > L:
        raise ValueError(f"program expands to {len(s)} instructions; the machine holds {L - 1}")
    ops = [OP[c] for c in s] + [OP["F"]] * (L - len(s))
    tgt = [-1] * L
    stack = []
    for i, c in enumerate(s):
        if c == "[":
            stack.append(i)
        elif c == "]":
            if stack:
                j = stack.pop()
                tgt[j], tgt[i] = i + 1, j + 1
            else:
                ops[i] = OP["NOP"]
    for j in stack:
        ops[j] = OP["NOP"]
    return ops, tgt


def run_reference(prog: str, inputs, N: int, T_out: int, budget: int, L: int):
    """The plain interpreter. Returns (output bytes, tape, dp, steps, stop reason)."""
    ops, tgt = compile_program(prog, L)
    K = len(inputs)
    tape, dp, ip, inp, out, steps = [0] * N, 0, 0, 0, [], 0
    while True:
        o = OPS[ops[ip]]
        if o == "F":
            return out, tape, dp, steps, "halt"
        if o == ">":
            dp = (dp + 1) % N; ip += 1
        elif o == "<":
            dp = (dp - 1) % N; ip += 1
        elif o == "+":
            tape[dp] = (tape[dp] + 1) % 256; ip += 1
        elif o == "-":
            tape[dp] = (tape[dp] - 1) % 256; ip += 1
        elif o == ".":
            out.append(tape[dp]); ip += 1
        elif o == ",":
            tape[dp] = int(inputs[inp]); inp = (inp + 1) % K; ip += 1
        elif o == "[":
            ip = tgt[ip] if tape[dp] == 0 else ip + 1
        elif o == "]":
            ip = tgt[ip] if tape[dp] != 0 else ip + 1
        else:                                                          # NOP
            ip += 1
        steps += 1
        if len(out) >= T_out:
            return out, tape, dp, steps, "output"
        if steps >= budget:
            return out, tape, dp, steps, "budget"


# ------------------------------------------------------------------------------------------------- the written machine
class WrittenBF:
    def __init__(self, L=128, N=16, K=32, S=30.0, M=20.0, n_head=4, dtype=torch.float64):
        t0 = time.time()
        self.L, self.N, self.K, self.S, self.M = L, N, K, S, M
        base = 0

        def take(w):
            nonlocal base
            sl = slice(base, base + w); base += w
            return sl

        def one():
            nonlocal base
            i = base; base += 1
            return i
        # token-class flags
        self.PROG, self.TAPE, self.INP, self.REG, self.NUL = one(), one(), one(), one(), one()
        # program tokens
        self.ppos, self.ins, self.jmp = take(L), take(10), take(L)
        # tape and input tokens (the byte subspace is shared: a token is one class or the other)
        self.tpos, self.kpos = take(N), take(K)
        self.bhi, self.blo = take(16), take(16)
        # register state
        self.ip, self.dp, self.inp = take(L), take(N), take(K)
        # scratch, cleared every pass
        self.r_ins, self.r_jmp = take(10), take(L)
        self.rv_hi, self.rv_lo, self.rin_hi, self.rin_lo = take(16), take(16), take(16), take(16)
        self.Z = one()
        self.nv_hi, self.nv_lo = take(16), take(16)
        self.wv_hi, self.wv_lo, self.here = take(16), take(16), one()
        self.out_hi, self.out_lo, self.emit, self.halt = take(16), take(16), one(), one()
        self.ip_nx, self.dp_nx, self.inp_nx = take(L), take(N), take(K)
        self.d_layout = base
        need_hd = max(L + 10, N + 1, K + 1, 33)
        hd = max(int(math.ceil(base / n_head)), need_hd)
        self.hd, self.n_head, self.d = hd, n_head, hd * n_head
        d = self.d
        self.A = H1.Block(d, n_head, "learned", norm="none")
        self.B = H1.Block(d, n_head, "learned", norm="none")
        for blk in (self.A, self.B):
            blk.attn.causal = False
            for p in blk.parameters():
                p.data.zero_()
        self.dtype = dtype
        self.A.to(dtype); self.B.to(dtype)
        with torch.no_grad():
            self._write_attention()
            self._write_mlp()
        self._write_boundary()
        self._sparse = None
        self.compile_seconds = time.time() - t0

    # ---- attention: four Match heads -------------------------------------------------------------------------------
    def _write_attention(self):
        d, hd, S = self.d, self.hd, self.S
        sq = S * math.sqrt(hd)                                         # SDPA divides q.k by sqrt(hd)

        def q(b, h, j, src, v): b.attn.qkv.weight[h * hd + j, src] = v
        def k(b, h, j, src, v): b.attn.qkv.weight[d + h * hd + j, src] = v
        def v_(b, h, j, src, v): b.attn.qkv.weight[2 * d + h * hd + j, src] = v
        def o(b, h, j, dst, v): b.attn.proj.weight[dst, h * hd + j] = v

        non_reg = (self.PROG, self.TAPE, self.INP, self.NUL)

        def match(b, h, qpos, kpos, width, null_q):
            """The register's (or a tape cell's) position code against the keys' position codes; every token whose
            query has no position mass goes to the NULL token instead, whose value is zero."""
            for j in range(width):
                q(b, h, j, qpos.start + j, sq)
                k(b, h, j, kpos.start + j, 1.0)
            for flag, strength in null_q:
                q(b, h, width, flag, strength * sq)
            k(b, h, width, self.NUL, 1.0)

        A, B = self.A, self.B
        # A.0 fetch: register <- prog[ip] : opcode and bracket target
        match(A, 0, self.ip, self.ppos, self.L, [(f, 1.0) for f in non_reg])
        for c in range(10):
            v_(A, 0, c, self.ins.start + c, 1.0); o(A, 0, c, self.r_ins.start + c, 1.0)
        for i in range(self.L):
            v_(A, 0, 10 + i, self.jmp.start + i, 1.0); o(A, 0, 10 + i, self.r_jmp.start + i, 1.0)
        # A.1 read: register <- tape[dp]
        match(A, 1, self.dp, self.tpos, self.N, [(f, 1.0) for f in non_reg])
        for c in range(16):
            v_(A, 1, c, self.bhi.start + c, 1.0); o(A, 1, c, self.rv_hi.start + c, 1.0)
            v_(A, 1, 16 + c, self.blo.start + c, 1.0); o(A, 1, 16 + c, self.rv_lo.start + c, 1.0)
        # A.2 inread: register <- input[inp]
        match(A, 2, self.inp, self.kpos, self.K, [(f, 1.0) for f in non_reg])
        for c in range(16):
            v_(A, 2, c, self.bhi.start + c, 1.0); o(A, 2, c, self.rin_hi.start + c, 1.0)
            v_(A, 2, 16 + c, self.blo.start + c, 1.0); o(A, 2, 16 + c, self.rin_lo.start + c, 1.0)
        # B.0 write: tape[tpos == dp] <- register's new value. A tape cell's null pull is HALF strength, so it takes
        # the register exactly when its position matches dp; every other token's pull is full strength.
        match(B, 0, self.tpos, self.dp, self.N,
              [(self.TAPE, 0.5)] + [(f, 1.0) for f in (self.PROG, self.INP, self.REG, self.NUL)])
        for c in range(16):
            v_(B, 0, c, self.nv_hi.start + c, 1.0); o(B, 0, c, self.wv_hi.start + c, 1.0)
            v_(B, 0, 16 + c, self.nv_lo.start + c, 1.0); o(B, 0, 16 + c, self.wv_lo.start + c, 1.0)
        v_(B, 0, 32, self.REG, 1.0); o(B, 0, 32, self.here, 1.0)        # "here": 1 iff the register was attended

    # ---- MLPs: threshold rows. GELU(M x)/M = relu(x) at integer x, so each row is relu(sum c_i x_i + bias) ---------------
    def _write_mlp(self):
        M = self.M
        I = {o: self.r_ins.start + OP[o] for o in OPS}

        class Rows:
            """Every row is GATED by the class of token it serves: one more literal (the class flag) and one more unit
            of threshold. On any other token the pre-activation is then -M, where GELU underflows to EXACTLY zero in
            float64. Without the gate a bias-0 row gives GELU(-eps) = -eps/2 on the wrong tokens from 1e-13 attention
            leakage, and BoundaryOp's quantise turns any nonzero group into a full one-hot: that is how every tape
            cell acquired a spurious dp = 0 in the first build, so that cell 0's write query matched N + 1 tokens."""

            def __init__(s, blk, gate):
                s.W1, s.b1, s.W2, s.r, s.gate = blk.mlp[0].weight, blk.mlp[0].bias, blk.mlp[2].weight, 0, gate

            def __call__(s, terms, bias, outs, gate=None):
                g = s.gate if gate is None else gate
                for dim, c in terms.items():
                    s.W1[s.r, dim] += M * c
                s.W1[s.r, g] += M
                s.b1[s.r] = M * (bias - 1)
                for dim, c in outs.items():
                    s.W2[dim, s.r] += c / M
                s.r += 1

        a = Rows(self.A, self.REG)
        # zero test of the current cell: both nibbles 0
        a({self.rv_lo.start: 1, self.rv_hi.start: 1}, -1, {self.Z: 1})
        for v in range(16):
            nl, nh = self.nv_lo.start + v, self.nv_hi.start + v
            lo = lambda u: self.rv_lo.start + (u % 16)                    # noqa: E731
            hi = lambda u: self.rv_hi.start + (u % 16)                    # noqa: E731
            # low nibble: +1, -1, set from input, or unchanged
            a({lo(v - 1): 1, I["+"]: 1}, -1, {nl: 1})
            a({lo(v + 1): 1, I["-"]: 1}, -1, {nl: 1})
            a({self.rin_lo.start + v: 1, I[","]: 1}, -1, {nl: 1})
            a({lo(v): 1, I["+"]: -1, I["-"]: -1, I[","]: -1}, 0, {nl: 1})
            # high nibble: the carry out of 15 and the borrow out of 0 are conjunctions -- one row each
            a({hi(v - 1): 1, I["+"]: 1, self.rv_lo.start + 15: 1}, -2, {nh: 1})
            a({hi(v): 1, I["+"]: 1, self.rv_lo.start + 15: -1}, -1, {nh: 1})
            a({hi(v + 1): 1, I["-"]: 1, self.rv_lo.start + 0: 1}, -2, {nh: 1})
            a({hi(v): 1, I["-"]: 1, self.rv_lo.start + 0: -1}, -1, {nh: 1})
            a({self.rin_hi.start + v: 1, I[","]: 1}, -1, {nh: 1})
            a({hi(v): 1, I["+"]: -1, I["-"]: -1, I[","]: -1}, 0, {nh: 1})
            # output: the current cell when the instruction is '.'
            a({self.rv_hi.start + v: 1, I["."]: 1}, -1, {self.out_hi.start + v: 1})
            a({self.rv_lo.start + v: 1, I["."]: 1}, -1, {self.out_lo.start + v: 1})
        a({I["."]: 1}, 0, {self.emit: 1})
        a({I["F"]: 1}, 0, {self.halt: 1})
        self.rows_A = a.r

        b = Rows(self.B, self.REG)
        L, N, K = self.L, self.N, self.K
        for i in range(L):
            dst = {self.ip_nx.start + i: 1}
            if i >= 1:
                prev = self.ip.start + i - 1
                b({prev: 1, I["["]: -1, I["]"]: -1}, 0, dst)          # not a bracket: step
                b({prev: 1, I["["]: 1, self.Z: -1}, -1, dst)          # '[' on a nonzero cell: step
                b({prev: 1, I["]"]: 1, self.Z: 1}, -2, dst)           # ']' on a zero cell: step
            b({self.r_jmp.start + i: 1, I["["]: 1, self.Z: 1}, -2, dst)  # '[' on zero: jump past the partner
            b({self.r_jmp.start + i: 1, I["]"]: 1, self.Z: -1}, -1, dst)  # ']' on nonzero: jump back
        for i in range(N):
            dst = {self.dp_nx.start + i: 1}
            b({self.dp.start + (i - 1) % N: 1, I[">"]: 1}, -1, dst)
            b({self.dp.start + (i + 1) % N: 1, I["<"]: 1}, -1, dst)
            b({self.dp.start + i: 1, I[">"]: -1, I["<"]: -1}, 0, dst)
        for i in range(K):
            dst = {self.inp_nx.start + i: 1}
            b({self.inp.start + (i - 1) % K: 1, I[","]: 1}, -1, dst)
            b({self.inp.start + i: 1, I[","]: -1}, 0, dst)
        for byte, wv in ((self.bhi, self.wv_hi), (self.blo, self.wv_lo)):
            for c in range(16):                                        # the pointed cell takes the new value
                b({wv.start + c: 1, self.here: 1}, -1, {byte.start + c: 1}, gate=self.TAPE)
                b({byte.start + c: 1, self.here: 1}, -1, {byte.start + c: -1}, gate=self.TAPE)
        self.rows_B = b.r
        if max(self.rows_A, self.rows_B) > 4 * self.d:
            raise ValueError("more threshold rows than the block's 4d hidden units")

    # ---- the boundary operator: the substrate's own module -------------------------------------------------------------
    def _write_boundary(self):
        d = self.d

        def read(src):
            m = torch.zeros(d, src.stop - src.start)
            for j in range(src.stop - src.start):
                m[src.start + j, j] = 1.0
            return m

        scratch = [self.r_ins, self.r_jmp, self.rv_hi, self.rv_lo, self.rin_hi, self.rin_lo, self.nv_hi, self.nv_lo,
                   self.wv_hi, self.wv_lo, self.out_hi, self.out_lo, self.ip_nx, self.dp_nx, self.inp_nx]
        keep = torch.ones(d)
        for sl in scratch:
            keep[sl] = 0.0
        for dim in (self.Z, self.here, self.emit, self.halt):
            keep[dim] = 0.0
        self.boundary = H1.BoundaryOp(
            keep,
            quantise=[(read(self.ip_nx), self.ip), (read(self.dp_nx), self.dp), (read(self.inp_nx), self.inp),
                      (read(self.bhi), self.bhi), (read(self.blo), self.blo)],
            halt_flag=self.halt, tie_tol=1e-3).to(self.dtype)

    # ---- encoding ------------------------------------------------------------------------------------------------------
    def encode(self, prog: str, inputs):
        ops, tgt = compile_program(prog, self.L)
        L, N, K = self.L, self.N, self.K
        T = L + N + K + 2
        x = torch.zeros(1, T, self.d, dtype=self.dtype)
        for i in range(L):
            x[0, i, self.PROG] = 1; x[0, i, self.ppos.start + i] = 1; x[0, i, self.ins.start + ops[i]] = 1
            if tgt[i] >= 0:
                x[0, i, self.jmp.start + tgt[i]] = 1
        for j in range(N):
            t = L + j
            x[0, t, self.TAPE] = 1; x[0, t, self.tpos.start + j] = 1
            x[0, t, self.bhi.start] = 1; x[0, t, self.blo.start] = 1
        for j in range(K):
            t = L + N + j
            b = int(inputs[j])
            x[0, t, self.INP] = 1; x[0, t, self.kpos.start + j] = 1
            x[0, t, self.bhi.start + (b >> 4)] = 1; x[0, t, self.blo.start + (b & 15)] = 1
        r = L + N + K
        x[0, r, self.REG] = 1; x[0, r, self.ip.start] = 1; x[0, r, self.dp.start] = 1; x[0, r, self.inp.start] = 1
        x[0, r + 1, self.NUL] = 1
        self.reg = r
        return x

    # ---- the forward pass: dense = the literal h1_lid.Block.forward; sparse = the same tensors, a sparse kernel --------
    def _sparse_blocks(self):
        if self._sparse is None:
            sp = []
            for blk in (self.A, self.B):
                sp.append({
                    "qkv": blk.attn.qkv.weight.detach().to_sparse_csr(), "qkv_b": blk.attn.qkv.bias.detach(),
                    "proj": blk.attn.proj.weight.detach().to_sparse_csr(), "proj_b": blk.attn.proj.bias.detach(),
                    "w1": blk.mlp[0].weight.detach().to_sparse_csr(), "b1": blk.mlp[0].bias.detach(),
                    "w2": blk.mlp[2].weight.detach().to_sparse_csr(), "b2": blk.mlp[2].bias.detach()})
            self._sparse = sp
        return self._sparse

    def _block_sparse(self, p, x):
        X = x[0]                                                       # (T, d)
        T, d, h, hd = X.shape[0], self.d, self.n_head, self.hd
        qkv = (p["qkv"] @ X.T).T + p["qkv_b"]
        q, k, v = qkv.split(d, dim=-1)
        q, k, v = (z.view(T, h, hd).transpose(0, 1) for z in (q, k, v))
        att = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(hd), dim=-1)
        y = (att @ v).transpose(0, 1).reshape(T, d)
        X = X + (p["proj"] @ y.T).T + p["proj_b"]
        hid = F.gelu((p["w1"] @ X.T).T + p["b1"])
        X = X + (p["w2"] @ hid.T).T + p["b2"]
        return X[None]

    def step(self, x, sparse=True):
        with torch.no_grad():
            if sparse:
                sp = self._sparse_blocks()
                x = self._block_sparse(sp[0], x)
                return self._block_sparse(sp[1], x)
            return self.B(self.A(x))

    def run(self, prog: str, inputs, T_out: int, budget: int, sparse=True, trace=None):
        x = self.encode(prog, inputs)
        r = self.reg
        out, steps = [], 0
        with torch.no_grad():
            while True:
                x = self.step(x, sparse)
                if trace is not None:
                    trace.append(x.clone())
                reg = x[0, r]
                if reg[self.halt] > 0.5:
                    reason = "halt"
                    break
                if reg[self.emit] > 0.5:
                    out.append(int(reg[self.out_hi].argmax()) * 16 + int(reg[self.out_lo].argmax()))
                steps += 1
                stop = "output" if len(out) >= T_out else ("budget" if steps >= budget else None)
                x, _ = self.boundary(x)
                if stop:
                    reason = stop
                    break
        L, N = self.L, self.N
        tape = [int(x[0, L + j, self.bhi].argmax()) * 16 + int(x[0, L + j, self.blo].argmax()) for j in range(N)]
        dp = int(x[0, r, self.dp].argmax())
        return out, tape, dp, steps, reason

    def nonzeros(self):
        return {name: int((t != 0).sum()) for name, t in
                (("A.qkv", self.A.attn.qkv.weight), ("A.proj", self.A.attn.proj.weight),
                 ("A.mlp0", self.A.mlp[0].weight), ("A.mlp2", self.A.mlp[2].weight),
                 ("B.qkv", self.B.attn.qkv.weight), ("B.proj", self.B.attn.proj.weight),
                 ("B.mlp0", self.B.mlp[0].weight), ("B.mlp2", self.B.mlp[2].weight))}

    def n_params(self):
        return sum(p.numel() for blk in (self.A, self.B) for p in blk.parameters())


# ------------------------------------------------------------------------------------------------- programs to run
UNIT = [
    ("paper Appendix E example", "+++[>+.<-]F", [1, 2, 3]),
    ("increment and print", "+.", [1]),
    ("wrap below zero", "-.", [255]),
    ("wrap above 255", "-+.", [0]),                                   # 255 + 1 = 0: the carry out of the top nibble
    ("tape wraps left", "<+.>.", [1, 0]),
    ("nested loops 3x2", "+++[>++[>+<-]<-]>>.", [6]),
    ("input echo", ",.,.,.", None),
    ("unmatched ']' is a no-op", "+].", [1]),
    ("unmatched '[' is a no-op", "[+.", [1]),
    ("loop never entered", "[+++.]+.", [1]),
    ("macro Z clears", "+++Z.", [0]),
    ("macro R moves right", "+++R>.", [3]),
    ("macro L triples right", "++L>.", [6]),
    ("macro N subtracts left", "+++++>++N<.", [3]),
    ("macro C copies to two", "++++C>.>.", [4, 4]),
    ("macro G scans right", "+>+>+>>>+<<<<<G.", None),
    ("macro H scans left", ">>>+<+<+H.", None),
    ("macro W", "+++W>.", [1]),
    ("macro V prints a string", "+>++>+++<<V", [1, 2, 3]),
    ("macro X sets 16", "X.", [16]),
    ("F halts early", "+.F+.", [1]),
    ("a jump over F", "[F]+.", [1]),
    ("carry chain 15 -> 16", "+" * 15 + ".+.", [15, 16]),
    ("borrow 16 -> 15", "X.-.", [16, 15]),
]


def sample_uniform(rng, max_len):
    """The paper's `uniform` ablation distribution (Table 5): program tokens i.i.d. uniform over the alphabet."""
    n = int(rng.integers(1, max_len + 1))
    return "".join(rng.choice(list(ALPHABET), size=n))


def sample_loopy(rng, max_len, depth=3):
    """Pure Brainfuck with balanced brackets and no early F, to exercise loops, nesting and long runs."""
    out, open_ = [], 0
    n = int(rng.integers(4, max_len + 1))
    for _ in range(n):
        r = rng.random()
        if r < 0.12 and open_ < depth:
            out.append("["); open_ += 1
        elif r < 0.24 and open_ > 0:
            out.append("]"); open_ -= 1
        else:
            out.append(str(rng.choice(list("><+-.,"), p=[.17, .17, .25, .2, .13, .08])))
    out += ["]"] * open_
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--T_out", type=int, default=32)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--n_uniform", type=int, default=400)
    ap.add_argument("--n_loopy", type=int, default=200)
    ap.add_argument("--n_dense", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    m = WrittenBF(args.L, args.N, args.K)
    rep = {"experiment": "E46", "L": args.L, "N": args.N, "K": args.K, "T_out": args.T_out, "budget": args.budget,
           "d_layout": m.d_layout, "d": m.d, "n_head": m.n_head, "hd": m.hd, "params": m.n_params(),
           "nonzeros": m.nonzeros(), "rows": {"A": m.rows_A, "B": m.rows_B}, "S": m.S, "M": m.M,
           "compile_seconds": round(m.compile_seconds, 3)}
    print(f"compiled: d={m.d} (layout {m.d_layout}), {m.n_head} heads of {m.hd}, rows A {m.rows_A} / B {m.rows_B}, "
          f"{sum(rep['nonzeros'].values())} nonzeros of {rep['params']:,} parameters, {m.compile_seconds:.2f}s",
          flush=True)

    def judge(name, prog, inputs, expect=None, sparse=True):
        ref = run_reference(prog, inputs, args.N, args.T_out, args.budget, args.L)
        got = m.run(prog, inputs, args.T_out, args.budget, sparse=sparse)
        same = (ref[0] == got[0]) and (ref[1] == got[1]) and (ref[2] == got[2]) and (ref[3] == got[3]) \
            and (ref[4] == got[4])
        rec = {"name": name, "prog": prog, "expanded_len": len(expand(prog)), "steps": ref[3], "stop": ref[4],
               "out_len": len(ref[0]), "exact": bool(same)}
        if expect is not None:
            rec["matches_expected"] = ref[0] == expect
        if not same:
            rec.update(ref_out=ref[0], got_out=got[0], ref_tape=ref[1], got_tape=got[1], ref_dp=ref[2], got_dp=got[2],
                       ref_steps=ref[3], got_steps=got[3], ref_stop=ref[4], got_stop=got[4])
        return rec

    # 1. unit programs, each instruction and macro, with hand-computed expectations where they are easy
    unit = []
    for name, prog, expect in UNIT:
        inputs = rng.integers(0, 256, args.K)
        unit.append(judge(name, prog, inputs, expect))
    rep["unit"] = unit
    bad_expect = [u for u in unit if u.get("matches_expected") is False]
    print(f"unit: {sum(u['exact'] for u in unit)}/{len(unit)} exact against the reference; "
          f"reference matches the hand-computed output on {sum(u.get('matches_expected', True) for u in unit)}"
          f"/{len(unit)}", flush=True)
    for u in unit:
        if not u["exact"] or u.get("matches_expected") is False:
            print("   MISS", json.dumps(u), flush=True)

    # 2. dense == sparse: the literal h1_lid.Block.forward on a subset, against the sparse kernel used for the sweep
    dense_rows, dense_ok = [], 0
    pool = [p for _, p, _ in UNIT[:6]] + [sample_loopy(rng, 30) for _ in range(args.n_dense)]
    for prog in pool:
        inputs = rng.integers(0, 256, args.K)
        a = m.run(prog, inputs, args.T_out, args.budget, sparse=False)
        b = m.run(prog, inputs, args.T_out, args.budget, sparse=True)
        dense_ok += int(a == b)
        dense_rows.append({"prog": prog, "same": a == b, "steps": a[3]})
    rep["dense_vs_sparse"] = {"n": len(pool), "identical": dense_ok}
    print(f"dense h1_lid.Block.forward vs sparse kernel: {dense_ok}/{len(pool)} identical runs", flush=True)

    # 3. the sweep
    def sweep(kind, sampler, n, max_len):
        rows, skipped = [], 0
        while len(rows) < n:
            prog = sampler(rng, max_len)
            if len(expand(prog)) + 1 > args.L:
                skipped += 1
                continue
            rows.append(judge(kind, prog, rng.integers(0, 256, args.K)))
        return rows, skipped

    t1 = time.time()
    uni, sk_u = sweep("uniform", sample_uniform, args.n_uniform, 24)
    loopy, sk_l = sweep("loopy", sample_loopy, args.n_loopy, 48)
    rep["sweep_seconds"] = round(time.time() - t1, 1)
    for kind, rows, sk in (("uniform", uni, sk_u), ("loopy", loopy, sk_l)):
        ex = sum(r["exact"] for r in rows)
        stops = {s: sum(r["stop"] == s for r in rows) for s in ("halt", "output", "budget")}
        rep[kind] = {"n": len(rows), "exact": ex, "skipped_too_long": sk, "stops": stops,
                     "mean_steps": float(np.mean([r["steps"] for r in rows])),
                     "max_steps": int(max(r["steps"] for r in rows)),
                     "mean_out_len": float(np.mean([r["out_len"] for r in rows])),
                     "mean_expanded_len": float(np.mean([r["expanded_len"] for r in rows])),
                     "misses": [r for r in rows if not r["exact"]][:10]}
        print(f"{kind:>8}: {ex}/{len(rows)} exact | stops {stops} | mean steps {rep[kind]['mean_steps']:.1f} "
              f"(max {rep[kind]['max_steps']}) | mean expanded length {rep[kind]['mean_expanded_len']:.1f} | "
              f"skipped {sk} too long", flush=True)
    total = len(unit) + len(uni) + len(loopy)
    exact = sum(u["exact"] for u in unit) + rep["uniform"]["exact"] + rep["loopy"]["exact"]
    steps_total = sum(r["steps"] for r in uni + loopy + unit)
    rep["verdict"] = {
        "exact": exact, "total": total, "rate": exact / total,
        "dense_equals_sparse": dense_ok == len(pool),
        "pass": exact / total >= 0.99 and dense_ok == len(pool),
        "passes_executed": steps_total,
        "instructions_used": ["Match (4 heads: fetch, read, inread, write)", "Row (threshold units, signed keys)",
                              "Quantise", "Keep/Clear", "Halt", "Readout"],
        "instructions_not_used": ["Gather", "Pool", "Broadcast", "Compare", "Branch"],
    }
    rep["seconds"] = round(time.time() - t0, 1)
    d = HERE / "runs" / "e46"; d.mkdir(parents=True, exist_ok=True)
    (d / "e46.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("\nverdict:", json.dumps(rep["verdict"]), f"{rep['seconds']}s")


if __name__ == "__main__":
    main()
