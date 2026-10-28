# The 25 public ARC-AGI-3 games: mechanics catalogue

Source: `environment_files/<game>/<version>/<game>.py` (subclasses of `arcengine.ARCBaseGame`), read 2026-10-27.
Purpose: list the general abilities one agent must learn. This file describes mechanics only. It holds no
solutions, strategies or action sequences.

Conventions. Colours are palette indices 0-15. Coordinates are (x, y) in the 64x64 output frame, origin top-left.
"Budget" means the per-level counter that ends the level in GAME_OVER when exhausted. "Baseline" is the human
action count per level from `metadata.json`.

## Engine facts that hold for every game

- Turn-based. Nothing moves between actions. One action returns a list of 1..N frames (the engine calls `step()`
  until the game marks the action complete). Intermediate frames are animation; the last frame is the settled state.
- `set_level()` resets the engine's action counter, so every budget below is per level. Retrying a level after
  GAME_OVER or RESET refills it.
- RESET (action 0): if at least one action was taken in the current level, it restarts that level. If no action
  was taken since the level began (including right after a level-up, or a second RESET in a row), it restarts the
  WHOLE GAME at level 1 and sets `levels_completed` to 0.
- After GAME_OVER the game accepts only RESET, which restarts the current level (levels already completed stay).
- ACTION6 coordinates are 64x64 display pixels. Each game maps them to its own grid (`display_to_grid`). A small
  grid (for example 16x16) is upscaled, and letterbox margins are drawn in a padding colour. Clicks in the margin
  are ignored.
- Every one of the 25 games draws a budget indicator inside the 64x64 frame (a bar, pips or a sliding sprite).
  It is part of the frame and changes on every action.

---

## ar25 (8 levels, keys + click, actions 1-7; baseline 32,50,75,37,89,159,233,73)
- **Controls:** No avatar. A 21x21 grid (x3) holds movable shapes and one or two mirror AXES (a vertical and/or a
  horizontal line). ACTION1-4 move the selected object 1 cell (a vertical axis moves only left/right, a horizontal
  axis only up/down). ACTION5 cycles the selection. ACTION6 selects the clicked shape or axis. ACTION7 = undo.
- **Display:** each shape is drawn together with its reflections across the axes (reflections of reflections too,
  up to depth 12); reflected copies are colour 4. Some shapes reflect in one axis only. Some shapes rotate 90 deg
  whenever their distance to an axis changes.
- **Win:** every small target dot is covered by a shape or by a reflection of a shape. Dots visible; the rule that
  reflections count must be inferred.
- **Loss:** budget 64,64,128,128,128,320,320,320. Moves and ACTION5 cost 1; clicks and undo are free. Vertical bar
  on column 63 that changes colour every 64 units (11, 12, 15, 8, 14).
- **Autonomous/frames:** none. L2 shows a blinking hint over a shape after 14 moves without a selection change.
- **Levels:** more shapes, both axes, one-axis and rotating shapes.
- **Abilities:** symmetry reasoning (compose reflections); move the transform as well as the object; select-then-move;
  coverage planning; undo; budget.

## bp35 (9 levels, keys + click, actions 3,4,6,7; baseline 21,48,44,38,33,87,86,131,163)
- **Controls:** Avatar in a side-view platformer (6-px cells); the view scrolls vertically to follow it. ACTION3/4
  step one cell left/right (bump animation if blocked); stepping off a ledge makes the avatar fall until it lands.
  No jump. ACTION6 on special blocks: breakable block disappears; growing block sprouts into all empty neighbour
  cells; toggle block switches solid/passable; gravity block flips gravity. The avatar falls again if its support
  changed. ACTION7 = undo.
- **Win:** the avatar touches or lands on the gem. Visible, possibly off-screen until the view scrolls.
- **Loss:** landing on spikes (death animation); L1-3: a spiked ceiling descends one cell every 2 actions and kills
  the avatar when it reaches the avatar's row. Budget 64 actions (L1-6) / 128 (L7-9); every action counts, undo too.
  Bar on row 63 (a second colour layer after 64).
- **Autonomous/frames:** the descending ceiling (L1-3). Custom renderer: each action returns an animation sequence
  (falls, crumbling, camera pans).
- **Levels:** L1-2 breakable blocks and spikes; L3 adds toggle blocks; L4+ gravity blocks; L8-9 growing blocks.
- Maps are stored y-reversed in the source; I read gravity as initially pointing to the bottom of the screen
  (unclear from source).
- **Abilities:** support/gravity physics; terrain editing by click to create paths; gravity flip; scrolling view;
  time pressure; undo; budget.

## cd82 (6 levels, keys + click, actions 1-6; baseline 55,8,41,21,23,23)
- **Controls:** No avatar. A paint bucket sits in one of 8 slots around a 10x10 canvas at (27,34). ACTION1-4 move the
  bucket around the 3x3 ring (centre excluded). ACTION5 tips the bucket (about 14 frames): it paints a half of the
  canvas (side slot) or a triangular half (corner slot) in the current colour. ACTION6 on a colour swatch in the top
  row (y~2) selects the colour (a marker moves under it; the bucket is recoloured). From L3 a small nozzle appears
  beside the bucket at side slots; clicking it paints a 3x4 strip in the middle of that side.
- **Win:** canvas equals the target picture drawn at top-left (3,3), both diagonals ignored. Template on screen.
- **Loss:** budget 100 actions; bar on row 63 (colour 4 remaining, 5 used).
- **Autonomous/frames:** paint and nozzle animations (multi-frame).
- **Levels:** L1 2 swatches, L2 3, L3-6 7 swatches plus nozzle; targets need more layers.
- **Abilities:** copy a displayed template; decompose it into overlapping fills where later paint overwrites earlier
  (order planning); map slot -> canvas region; select colour by click, apply by key; budget.

## cn04 (6 levels, keys + click, actions 1-6; baseline 29,54,85,300,208,113)
- **Controls:** No avatar. 2-8 coloured outline pieces on a 20x20 board (x3). Each piece has "pin" pixels (colour 8).
  ACTION6 selects a piece (its body turns white 0; the piece nearest the origin starts selected); clicking it again
  deselects. ACTION1-4 move the selected piece 1 cell (clamped to the board). ACTION5 rotates it 90 deg; where
  several alternative shapes are stacked at one spot (L5-6), ACTION5 cycles to the next shape instead.
- **Win:** every pin of every visible piece coincides with exactly one pin of another piece. Matched pins turn
  colour 3 (partial-progress feedback). No template.
- **Loss:** budget 75,100,125,125,150,200; bar on row 0, x=16..47 (colour 4 remaining, 0 used from the left).
- **Autonomous/frames:** none.
- **Levels:** from L3 "grey masking": unselected pieces are drawn all colour 4 except matched pins, so pin layouts are
  visible only while a piece is selected. Background colour changes per level.
- **Abilities:** jigsaw/connector matching under rotation; select-then-transform; choose among shape variants;
  memory under masked observation; budget.

## dc22 (6 levels, keys + click, actions 1,2,3,4,6; baseline 59,102,67,98,324,578)
- **Controls:** Avatar (small sprite). ACTION1-4 move it 2 px; it must stay on floor/platform sprites (moves off the
  floor are refused). A control panel on the right holds coloured buttons. ACTION6 on a button toggles every object
  with that button's colour code to its alternate form (platforms and bridges appear, vanish or change shape; the
  avatar rides along). Later levels add a crane on a 4x4 lattice driven by four arrow buttons, plus a grab button
  that attaches a bridge piece or object to the crane; an illegal crane move shakes it.
- **Win:** the avatar's position equals the goal sprite's position. Visible.
- **Loss:** budget 128,192,192,192,512,1024; every action costs 1; bar on row 63 (colour 0 remaining, 3 used). If a
  click leaves the avatar without floor, a 14-frame fall plays (avatar fills grey, the screen closes to black in a
  shrinking circle), the board is restored to before the click and 20 budget units are deducted.
- **Autonomous/frames:** fall and crane animations.
- **Levels:** some buttons stay hidden until the avatar stands on a matching key zone or touches a matching pickup;
  crane levels; grid 64x44 / 64x48 in early levels (letterbox rows).
- **Abilities:** discover remote cause-effect links (button colour -> object set); navigation on a changing map;
  tool use (crane carries a bridge); conditional affordances; hazard = losing support; budget.

## ft09 (6 levels, click only, action 6; baseline 43,12,23,28,65,37)
- **Controls:** No avatar. A 32x32 grid (x2) with a lattice of 3x3 tiles on a 4-px pitch. Clicking a tile cycles its
  colour through a per-level palette of 2-3 colours (9/8, 9/12, 8/12, 9/8/12, 14/15, 11/14). Special tiles (centre 10
  with colour-6 marks; L5+) cycle the tiles at the marked neighbour offsets (a plus shape) instead of themselves.
- **Win:** every constraint key holds at once. A key is a 3x3 sprite inside the lattice: its centre pixel is a
  reference colour; each border pixel states whether the neighbouring tile in that direction must EQUAL (border
  pixel 0) or DIFFER from the reference colour. The rule is displayed but must be decoded.
- **Loss:** click budget 32,32,96,96,128,128; only clicks on tiles count; bar on row 63 (colour 12 remaining, 11 used).
- **Autonomous/frames:** L1 contains decorative/tutorial sprites; clicking empty space in L1 makes one blink.
- **Levels:** larger lattices, 3-colour cycles, coupled (plus-pattern) tiles.
- **Abilities:** decode a constraint legend; constraint satisfaction on a grid; colour cycle order (mod n); coupled
  toggles (Lights-Out style linear algebra); budget.

## g50t (7 levels, keys, actions 1-5; baseline 78,175,179,230,96,54,67)
- **Controls:** Avatar: 7x7 colour-9 square, moves 6 px per key along a drawn path; cannot leave the path or enter a
  closed door. ACTION5 = rewind/loop: the run is rewound (reverse animation), the avatar restarts at the start, and
  the previous run becomes a grey GHOST that replays its recorded moves in lock-step with the new moves. Small
  loop-slot sprites at the top-left (2-3 per level); using all of them clears the ghosts.
- **Win:** the avatar reaches the exit frame. Visible.
- **Loss:** a timer bar on row 63 (64 px, colour 9) slides left 1 px every 2 actions; GAME_OVER once it has left the
  screen (about 130 actions). The avatar crushed by a door or caught by an enemy (shatter animation) -> GAME_OVER.
- **Autonomous/frames:** patrolling enemies step along tracks once per move; ghosts move with you; moves slide
  1 px/frame (6 frames); door, rewind and death animations.
- **Levels:** pressure plates wired to sliding doors (some toggle), pads with a swapper that exchanges whatever
  stands on two pads, enemies. Ghosts also press plates.
- **Abilities:** cooperate with your own past runs (plan run 1 for the benefit of run 2); remember action sequences;
  discover switch-door wiring; enemy timing; detect an on-screen timer.

## ka59 (7 levels, keys + click, actions 1,2,3,4,6; baseline 28,109,51,51,33,132,326)
- **Controls:** Several controllable blocks (outlined in colour 14; the selected one has a colour-0 centre, the others
  colour 4). ACTION6 selects a block (costs 1). ACTION1-4 move the selected block 3 px; walls and pits stop it.
  Pushing another block or a crate KICKS it: the pushed object slides 3 px per frame for at least 5 frames until a
  wall stops it; it can slide over pits but not stop on one; pushes chain.
- **Win:** every block-shaped target frame contains a controllable block of exactly that size, and every crate frame
  contains a crate. Targets visible as outlines.
- **Loss:** budget 100,127,100,127,100,150,200; bar on row 63 (colour 4 remaining, 0 used).
- **Autonomous/frames:** bombs (L5+): each action fills one more row of a bomb with colour 12 (a visible fuse); when
  full it explodes in its facing direction for 3 frames and shoves everything hit. Slides animate over frames.
- **Levels:** L1 two blocks; L2 blocks of different shapes; L3+ crates; L5+ bombs.
- **Abilities:** momentum/sliding physics; one-way obstacles (pits); choose which body to move; size/shape matching
  to frames; timed tools; dead ends (objects stuck against walls); budget.

## lf52 (10 levels, click (+keys), actions 1,2,3,4,6,7; baseline 32,81,60,71,205,148,244,109,164,225)
- **Controls:** No avatar. PEG SOLITAIRE on a board of 6-px cells (pegs are green discs). ACTION6 on a peg shows up
  to 4 jump markers two cells away; ACTION6 on a marker performs the jump; a jumped peg of the same colour is
  removed. Red/blue pegs (L6+) are hopped over but not removed; fixed posts can be jumped but never removed.
  ACTION1-4 slide every movable cart one cell along its rail (pegs on carts ride along). ACTION7 = undo.
- **Win:** one peg left (two in L6-7). Not displayed; the classic rule must be inferred.
- **Loss:** an internal counter of moves and clicks (undo not included): 64 in L1, 320 in L2-5, 640 in L6-10. The bar
  on row 0 shows a separate count of all actions (undo included), its colour changing every 64 actions, so the bar and
  the real limit can drift apart.
- **Dead ends:** at known dead positions all pegs turn grey and shake and a restart button slides in at bottom-left;
  clicking the bottom-left 16x16 corner then restarts the level (dissolve animation).
- **Autonomous/frames:** custom renderer: every action returns an animation sequence; the board is larger than the
  screen in later levels and the view pans (scripted) when a peg lands on certain cells. L1: clicking empty space
  blinks a hint sprite.
- **Abilities:** irreversible combinatorial search with lookahead; two-step click selection; transport on rails;
  scrolling view; undo; budget.

## lp85 (8 levels, click only, action 6; baseline 17,38,31,16,41,60,26,159)
- **Controls:** No avatar. Tiles on a 3-px lattice lie on one or more closed loops (rings) that overlap. Each loop has
  two buttons (one per direction); ACTION6 on a button shifts every tile on that loop one position along it (a cyclic
  permutation). Overlapping loops share tiles.
- **Win:** every type-A special tile sits on an A goal marker and every type-B tile on a B goal marker. Markers visible.
- **Loss:** budget 13,60,80,150,80,80,80,80; only button clicks count. Vertical bar on column 0 (colour 5 used,
  growing from the top; 14 remaining). Level-progress pips on row 1 (x = 10 + 5i, 4 px wide; 14 reached, 5 not yet).
- **Autonomous/frames:** none.
- **Levels:** more loops, more overlap, asymmetric loops; grid size varies (27x32 .. 63x63, letterboxed).
- **Abilities:** learn button -> permutation; group-style planning (commutators, avoid undoing one loop with another);
  multi-object assignment; budget.

## ls20 (7 levels, keys, actions 1-4; baseline 22,123,73,84,96,192,186)
- **Controls:** Avatar: 5x5 block (top rows colour 12, bottom 9), moves one 5-px cell per key; walls block. The avatar
  carries a KEY whose shape (6 kinds), colour (12/9/14/8) and rotation (4) are shown magnified in the bottom-left HUD
  (around x=3, y=55). Stepping on modifier tiles changes the key: one kind cycles shape, one cycles colour, one
  rotates it 90 deg. Some modifier tiles move along tracks one cell per avatar move. Push pads shove the avatar
  several cells (animated).
- **Win:** every door is opened. Each door displays the key it needs (shape + colour + rotation). Walking into a door
  with a non-matching key is refused (flash); with the matching key the door is consumed. L6 has two doors.
- **Loss:** 42 steps per life, shown as 1-px pips on rows 61-62 from x=13 (colour 11 left, 3 used); a move costs 1 or
  2 depending on the level. Running out costs a life: full-screen flash, level state restored, steps refilled. Three
  life pips at x=56,59,62, rows 61-62 (colour 8 = left); no lives = GAME_OVER. Pickups refill the steps.
- **Autonomous/frames:** moving modifier tiles; push animations. L7 has FOG: everything more than ~20 px from the
  avatar is drawn black (5). L1 only: markers light up when the key matches a door (tutorial hint).
- **Abilities:** an avatar with a visible attribute vector (inventory); compare it to a displayed template; modular
  counters (reach a value in the fewest cycles); route planning through modifiers; lives; memory under fog; budget.

## m0r0 (6 levels, keys + click, actions 1-6; baseline 30,111,203,26,500,237)
- **Controls:** Small grids (11-15 cells, upscaled). Two single-cell pieces (colour 10) move together with ACTION1-4;
  one of them is mirrored left-right (it moves -dx). Walls block each piece separately. ACTION6 on a movable block
  (colour 9) selects it (the pieces turn colour 1 = frozen) and ACTION1-4 then move the block; clicking elsewhere
  deselects. ACTION5 has no effect.
- **Win:** the two pieces meet (same cell, or swap through each other -> merged). Not displayed; must be inferred.
- **Hazards:** colour-8 cells (checker overlay): a piece landing on one triggers a 7-frame flash and returns all pieces
  and blocks to their start positions (soft reset, budget keeps running).
- **Loss:** budget 150; two bars, row 0 (shrinks from the right) and row 63 (shrinks from the left), colours 5/0.
- **Autonomous/frames:** none besides the hazard flash.
- **Levels:** L3+ movable blocks; L5-6 pressure plates (1 cell) and gates (3-cell bars): a piece on a plate removes all
  gates of the same colour while it stays there. Background colour-0 pixels are repainted with two per-level colours,
  left half / right half (mirrors the control symmetry).
- **Abilities:** coupled mirrored control of two bodies; use obstacles to desynchronise them; mode switch by click;
  plates and gates; hazard avoidance; budget.

## r11l (6 levels, click only, action 6; baseline 22,33,51,26,52,49)
- **Controls:** No avatar. Each "body" sits automatically at the centroid of its 2+ handles (small colour-3 markers;
  the selected handle is colour 0); thin colour-1 lines join each handle to its body. ACTION6 on a handle selects it;
  ACTION6 on any other point moves the selected handle there (refused if it would overlap a wall), and the body slides
  to the new centroid (animated).
- **Win:** every body overlaps its own target outline. Later levels: a body must first collect colour pickups by
  passing over them and counts only when its colour set equals its target's colour set. Targets visible.
- **Loss:** budget 60 actions; vertical bar on column 0 (colour 0 remaining, filled from the bottom; 5 used). A body
  touching a hazard blinks and the handle move is reversed; the 5th hazard hit in a level = GAME_OVER (count not shown).
- **Abilities:** indirect control through geometry (centroids); select-then-target clicking; paths around hazards and
  walls; collection sub-goals; budget.

## re86 (8 levels, keys, actions 1-5; baseline 26,42,86,108,189,139,424,241)
- **Controls:** No avatar. Several coloured line shapes (plus-crosses, hollow rectangles) on a 3-px lattice; the
  selected shape has a colour-0 dot at its centre. ACTION1-4 move it 3 px; ACTION5 cycles the selection. Shapes
  DEFORM against fixed obstacles: a cross pushed into an obstacle slides its cross-bar; a rectangle pushed into one is
  squashed (width -3 / height +3, or the reverse). Moving a shape onto a coloured pad recolours it (spreads over frames).
- **Win:** a full-screen target drawing (coloured outlines) is shown; every coloured target pixel must be covered by a
  shape pixel of the same colour (union of all shapes). Template on screen.
- **Loss:** budget 100,100,200,200,250,200,300,400; every action costs 1; bar on row 63 (colour 15 remaining, 1 used).
- **Levels:** more shapes, reshaping obstacles, recolouring pads.
- **Abilities:** template matching (shape + colour); cycle-select then move; learn non-rigid deformation from contact;
  recolour by location (possibly irreversible); order of operations; budget.

## s5i5 (8 levels, click only, action 6; baseline 20,89,106,54,162,38,86,83)
- **Controls:** No avatar. Coloured jointed ARMS (3-px-wide bars, a colour-3 cap marks the base) form chains; child
  arms ride on parents. Slider widgets with colour swatches: clicking the far half of a slider lengthens every arm of
  that colour by 3 px, the near half shortens it. Rotate buttons rotate every arm of their colour 90 deg about its
  base, children carried along.
- **Win:** every target marker has an end piece exactly on it. Targets visible.
- **Loss:** budget 50,150,200,100,150,150,200,200 clicks; every click counts, even on empty space; bar on row 63
  (colour 3 remaining, 4 used).
- **Autonomous/frames:** if arms overlap after a click, the illegal pose shows for one frame and is undone.
- **Abilities:** forward/inverse kinematics of linked bodies; colour-coded control-to-object binding; collision
  constraints; budget.

## sb26 (8 levels, keys + click, actions 5,6,7; baseline 18,28,18,19,31,23,58,18)
- **Controls:** No avatar. A visual PROGRAM: one or more frames (rows of empty slots), a palette of coloured tiles in the
  strip below row 53, and a target colour sequence along the top row (y~1). ACTION6: click a tile to pick it, then an
  empty slot to place it or another tile to swap with it. ACTION7 = undo. ACTION5 = RUN: a cursor steps through the
  first frame's slots; each colour tile must equal the next target cell (which then fills); a "call" tile (outlined)
  jumps into the frame of its colour, runs it, and returns. A mismatch or empty slot gives a red flash and aborts.
- **Win:** the run produces the whole target sequence in order. Template on screen.
- **Loss:** energy 64; each placement/swap costs 1 and each RUN costs 1; bar on row 53 (colour 2 remaining, 3 used).
- **Autonomous/frames:** RUN plays a long animation (about 6 frames per executed slot).
- **Levels:** L1 one frame (direct copy); L2+ several frames and call tiles, so a long target must be compressed into
  few slots.
- **Abilities:** program synthesis with subroutines (sequence compression); test by execution; pick-and-place by
  click; undo; budget.

## sc25 (6 levels, keys + click, actions 1,2,3,4,6; baseline 36,6,32,83,143,50)
- **Controls:** Avatar ("wizard"): ACTION1-4 move it 2 px (small) or 4 px (big) and set its facing. A 3x3 SPELL PAD at
  the bottom centre (cells near x=24/29/34, y=49/54/59): ACTION6 toggles a pad cell. When the lit cells form a known
  spell enabled in this level, it fires: 3-cell corner = teleport to the next teleport marker; 4-cell diamond = grow
  x2 / shrink (refused if no room); 3-cell vertical line = fireball in the facing direction that destroys a switch and
  thereby removes the matching barriers. Spell icons in a left panel: one click shows the pattern on the pad (colour 0
  cells), a second click plays a demonstration. L1-3 auto-play the level's spell demo on the first action.
- **Win:** walk into the exit sprite. Visible.
- **Loss:** one shared budget (50,25,50,35,65,60) for moves AND pad toggles; vertical bar at the right edge (x~62;
  colour 14 remaining, 0 used). Pickups give back 10.
- **Autonomous/frames:** spells, fireballs, teleports and resizing animate.
- **Levels:** L1-3 one spell each; L4-6 two or three spells to combine.
- **Abilities:** learn a gesture -> effect vocabulary from demonstrations; tool sequencing (switch -> barrier; shrink
  to pass; teleport across walls); facing; irreversible effects; budget shared by movement and casting.

## sk48 (8 levels, keys + click, actions 1,2,3,4,6,7; baseline 61,177,101,103,230,181,125,92)
- **Controls:** You drive an extendable ARM (head + chain of 6-px segments). The key in the head's facing direction
  extends the arm one segment, pushing coloured blocks ahead (chains push); the opposite key retracts one segment
  (blocks along the arm can be dragged); a perpendicular key slides the whole arm sideways only when the head sits on
  a rail. Walls and edges block. ACTION6 on a head (or on its reference copy) selects which arm is controlled.
  ACTION7 = undo.
- **Win:** for every arm, the colours of the blocks lying along it, in order from the head, equal the sequence of its
  reference arm drawn in the strip below row 53. Match markers appear on matched segments. Template on screen.
- **Loss:** budget 196 moves (only ACTION1-4 cost, undo does not refund); bar on row 53 (colour 2 remaining, 3 used),
  which also divides the play area from the reference strip.
- **Autonomous/frames:** each move animates 2 frames; level win plays a 35-frame flash.
- **Abilities:** sequence template matching; indirect manipulation with a tool (push and pull); rails; choose which
  body to control; undo; budget.

## sp80 (6 levels, keys + click, actions 1-6; baseline 39,58,25,148,96,152)
- **Controls:** No avatar. Sources at the top emit liquid when poured. Platforms (colour 8; selected 9) of width 3-8:
  ACTION6 selects one (the one nearest the origin starts selected); ACTION1-4 move it 1 cell (not into the top 3 rows,
  not touching a cup). ACTION5 = POUR: liquid advances one cell per frame; it splits to both ends of a platform it hits;
  diverters (colour 15, L5-6) turn it 90 deg; cups (colour 11) fill (-> 13) when liquid enters from above; edge strips
  (colour 1) on the floor and sides turn 14 when touched (spill).
- **Win:** after a pour, every cup is filled and nothing spilled. Inferred.
- **Loss:** budget 30,45,100,120,100,120 (each edit action costs 1); bar on row 0 (colour 14 remaining, 0 used). A
  failed pour flashes, clears the liquid and returns to editing; the 5th pour after 4 failures = GAME_OVER (failure
  count not shown).
- **Frame format:** in L2, L3 and L5 the WHOLE frame is rotated 180 deg (liquid falls upward on screen, the HUD bar
  appears on row 63); key and click mappings are remapped so on-screen directions stay consistent.
- **Abilities:** predict a physical simulation before committing (plan, then one irreversible run); flow splitting;
  select-then-move; view-rotation invariance; limited trials.

## su15 (9 levels, click + undo, actions 6,7; baseline 22,42,26,115,36,31,8,40,41)
- **Controls:** No avatar. Play area y=10..62. ACTION6 at a point emits a shrinking ring (radius 8, colour 0, 4 frames)
  that PULLS every item inside it toward the click point. Items have levels 0-8 (own colour/size each); two items of the
  same level that touch MERGE into one item of the next level (value doubles). Two touching items of different levels
  = error: they flash 16 frames, the move is undone, and a growing penalty (2, 4, 6 ... units) is charged. ACTION7 = undo.
- **Creatures (L3+):** three tiers; also pulled by the ring; otherwise they walk toward the nearest item and knock it
  down one level (a level-0 item is destroyed). Same-tier creatures merge upward; different tiers touching = error.
- **Win:** the goal zone(s) contain exactly the required items (e.g. one level-2 item; later several kinds including a
  creature tier). The requirement is drawn as item icons in the top strip (near x=30, y=2-4). Template on screen.
- **Dead-end signal:** when the total value of the remaining items can no longer reach the requirement, the goal zone
  and the top icons turn grey (colour 2); undo restores them.
- **Loss:** budget 32,32,48,48,32,32,32,48,48 clicks plus penalties; bar on row 63 (colour 0, length = remaining);
  also GAME_OVER when no items remain.
- **Abilities:** attraction physics (choose points that gather the right subset); powers-of-two arithmetic and
  counting; forbidden contacts; adversarial wanderers; reading goal icons; undo; budget.

## tn36 (7 levels, click only, action 6; baseline 32,72,26,40,30,55,62)
- **Controls:** No avatar. A board holds a coloured piece and a target outline; under it a PROGRAM panel of instruction
  slots, each a column of 2-6 bit toggles (set = colour 5, clear = 1). A slot's opcode is the binary number of its bits.
  ACTION6 on a bit toggles it; ACTION6 on the run button (or the board) runs the program, one instruction per frame.
  Opcodes move the piece (4 or 8 px in a direction), rotate it (90/180/270), scale it (+1/-1) or recolour it (three
  colours). Obstacles block a move; some barriers switch on/off every 3 instructions and destroy the piece if it is
  inside them; scale gates pass only a piece of matching scale. A failed run returns the piece to its start pose.
- **Examples:** from L2 a second board on the left with tabs; each tab loads a preset program and start pose (some
  read-only) and runs it, which reveals what opcodes do.
- **Win:** after a run, the right-hand piece equals the target in position, scale, rotation AND colour. Target visible.
- **Loss:** a timer bar from (1,1) at the top-left slides left 1 px per click (every 2nd click from L6); GAME_OVER when
  it has left its track. No other counter.
- **Abilities:** learn an instruction set (bit pattern -> transformation) from examples and experiments; program
  synthesis toward a target pose; binary encoding; timing of toggling hazards; budget.

## tr87 (6 levels, keys, actions 1-4; baseline 54,58,40,45,71,146)
- **Controls:** No avatar. Upper area: a table of rewrite RULES ("left symbol sequence -> right symbol sequence", an
  arrow sprite between). Lower area (below y~34): a SOURCE row of symbol tiles and an ANSWER row of editable tiles. Each
  symbol family has 7 variants. ACTION3/4 move a bracket cursor left/right over the editable tiles; ACTION1/2 cycle the
  selected tile to the previous/next variant.
- **Display:** every tile has a random (seeded) rotation; symbol identity ignores rotation.
- **Win:** the answer row equals the translation of the source row (parse the source left-to-right into rule left
  sides, concatenate the right sides). Rules shown; the answer must be computed. Success plays a colour-cycling
  animation over each matched pair (7 frames per pair).
- **Loss:** budget 128 (L1-5) / 256 (L6), every action counts; bar on row 63 (colour 1 remaining, 4 used).
- **Levels:** L1 one-to-one rules; L2-3 multi-symbol rules; L4 chained rules (a rule's output is looked up again via
  linked marker tiles); L5 the cursor edits the RULES so that the fixed source maps to the fixed answer; L6 combines
  chained translation, two-step translation and rule editing.
- **Abilities:** apply and induce a symbolic rewrite system; segmentation/parsing; inverse problems (edit rules to fit
  data); rotation-invariant symbol identity; cursor editing; budget.

## tu93 (9 levels, keys, actions 1-4; baseline 19,16,34,42,123,80,14,23,111)
- **Controls:** Avatar: 3x3 colour-9 block with a facing pixel (colour 4). A maze of colour-2 corridors on a 6-px
  lattice. ACTION1-4 move one cell if the corridor continues; otherwise the key is spent with no move.
- **Win:** reach the exit (3x3 colour 14).
- **Loss:** budget 50,50,35,20,50,60,30,50,50; every key press costs 1; bar on row 63 (colour 6 remaining, 0 used).
  Avatar destroyed (grow-and-vanish animation) -> GAME_OVER.
- **Enemies:** 3x3 colour 8 with a facing pixel 15 (turns 11 when active). Sentries activate when the avatar is exactly
  one cell in front of them and then charge forward each turn; patrollers move every turn and reverse at corridor
  ends; followers activate when the avatar is two cells in front and then replay the avatar's own move history.
  The avatar moving onto an enemy destroys it; an enemy moving onto the avatar destroys the avatar.
- **Autonomous/frames:** one action gives about 12 frames (the avatar slides 6 px, then the enemies slide).
- **Abilities:** maze navigation; identify agent types from behaviour; facing/line-of-sight; timing; predict an agent
  that copies you; budget.

## vc33 (7 levels, click only, action 6; baseline 7,18,44,61,131,34,152)
- **Controls:** No avatar. Columns stand side by side between walls along a per-level "gravity" axis (right, left,
  down or up; the whole layout is oriented accordingly). Clicking a pump button moves one unit of height from one
  column of a linked pair to the other (one rises, the other sinks), carrying the small coloured floaters on their
  tops. A gate between two columns lights up (colour 12, with an indicator) when both tops are flush; clicking a lit
  gate animates the floaters crossing over.
- **Win:** every floater sits at a target marker of its own colour beside the walls. Markers visible; the matching
  rule must be inferred.
- **Loss:** budget 50,50,75,50,200,50,200 clicks; every click counts, even on empty space; bar on row 0 (colour 7
  remaining, 4 used).
- **Levels:** grid 32..64 (letterboxed); more columns, pumps and floaters; the axis changes each level.
- **Abilities:** learn what each button does (remote binding); conserved quantities; conditional affordances (gate
  usable only when aligned); orientation invariance; multi-step planning; budget.

## wa30 (9 levels, keys, actions 1-5; baseline 71,119,183,98,368,68,79,442,415)
- **Controls:** Avatar on a 4-px lattice that turns to face its move direction. ACTION1-4 move one cell (walls, boxes
  and no-go cells block). ACTION5 grabs/releases the box directly in front (while held, avatar and box move rigidly
  together: push and pull; facing locked); ACTION5 facing a "thief" removes it.
- **Other agents (act after every action):** HELPERS walk to the nearest free box, grab it, carry it into the target
  zone and release it. THIEVES grab boxes and carry them into their own zone.
- **Win:** every box lies inside the target zone and nobody holds it. Zone visible; goal inferred.
- **Loss:** budget 200,70,100,100,125,75,125,150,70; every action costs 1; bar on row 63 (colour 7 remaining, 4 used).
- **Display:** a box outline shows its state (0 held by you, 3 in front of you, 5 held by another agent, 4 free); a
  thief outline is 11 when you face it, else 15.
- **Abilities:** Sokoban-style manipulation with grab/pull; model cooperative and adversarial agents and exploit or
  remove them; facing; budget.

---

## Summary

### General ability categories (with the games that need each)

Perception and representation
1. **Object segmentation and tracking across animation frames** (sprites, multi-cell bodies, settled vs intermediate
   frames): all 25.
2. **Detect and discount HUD elements** (budget bars, pips, lives, timers, level pips, loop slots): all 25.
3. **Transformation-invariant recognition** (rotation, reflection, whole-view rotation, layout re-orientation): tr87,
   sp80, vc33, ar25, cn04, tn36, ls20, ka59.
4. **Read a displayed target and compute the difference to the current state:** cd82, re86, sk48, sb26, su15, tn36,
   ls20, ka59 (exact templates); r11l, s5i5, lp85, vc33, ar25, wa30, sp80 (target markers/zones).
5. **Decode a displayed rule or legend** (not a picture of the goal): ft09 (constraint keys), tr87 (rewrite table),
   sc25 (spell icons), tn36 (example programs), ls20 (key icon on doors).
6. **Memory under partial observability:** ls20 (fog), cn04 (grey masking), lf52 and bp35 (scrolling view), g50t
   (recorded runs), tu93 (followers replay your history).

Action and control learning
7. **Learn an avatar's action model** (moves, facing, blocked moves, step size): tu93, wa30, ls20, g50t, dc22, sc25,
   bp35.
8. **Select a target object, then act on it** (click-select + keys, cycle-select key, or two-step clicks): cd82, cn04,
   ka59, m0r0, sp80, ar25, sk48, re86 (ACTION5 cycles), lf52, r11l, sb26.
9. **Coupled or indirect control** (one input moves several things, or moves a thing through intermediaries): m0r0,
   lp85, s5i5, vc33, r11l, dc22, lf52 (carts), ft09 (coupled tiles), sk48, cd82.
10. **Discover remote cause-effect bindings** (buttons, plates, switches, colour-coded links): m0r0, g50t, dc22, vc33,
    s5i5, lp85, sc25, ls20, tn36.
11. **Learn an action vocabulary or instruction set from demonstrations/experiment:** sc25 (spells), tn36 (opcodes),
    sb26 (call tiles), ls20 (modifier tiles).
12. **Use an undo action:** sk48, sb26, ar25, lf52, su15, bp35.
13. **Plan, then commit with one "run" action** (simulate before executing): sp80, sb26, tn36.

World dynamics
14. **Intuitive physics:** gravity/support (bp35, dc22), liquid flow and splitting (sp80), sliding momentum and chain
    pushes (ka59, sk48, wa30), attraction and merging (su15), conserved quantities (vc33), non-rigid deformation (re86),
    explosions (ka59), kinematic chains (s5i5).
15. **Model other agents' policies:** tu93 (sentry/patrol/follower), wa30 (helpers and thieves), g50t (patrols and your
    own ghosts), su15 (item-eating creatures).
16. **Modular / cyclic state:** ls20 (key shape/colour/rotation), ft09 (colour cycles), tr87 (7 variants), lp85
    (permutation cycles), s5i5 (4 rotations), tn36 (binary opcodes), ka59 (bomb fuse).

Goal inference and planning
17. **Infer the success rule when it is not stated:** no target shown at all in m0r0 (pieces meet), cn04 (all pins
    matched), lf52 (one peg left); target objects shown but the rule must be inferred in sp80 (all cups, no spill),
    wa30 (all boxes in zone), vc33 (colour-to-marker matching), ar25 (reflections count), ka59 (size-matched frames).
18. **Combinatorial search / constraint satisfaction:** ft09, cn04, lp85, lf52, ar25, su15, tr87 (L5-6).
19. **Symbolic and program induction:** tr87, sb26, tn36, sc25.
20. **Plan under irreversibility and dead ends:** lf52, su15, m0r0, ka59, tu93, bp35, sc25, ls20, g50t, sp80, wa30
    (thief removal), re86 (recolouring).
21. **Long-horizon planning with sub-goals:** ls20, g50t, dc22, sc25, sk48, wa30, cd82 (paint order), vc33, s5i5.
22. **Timing and temporal reasoning:** tu93, g50t (ghost sync, timer bar), ka59 (fuse), bp35 (descending ceiling),
    tn36 (barriers toggling every 3 instructions).
23. **Spatial reasoning:** pathfinding (tu93, wa30, ls20, g50t, dc22, sc25, bp35); geometry (r11l centroids, s5i5
    kinematics, ar25 symmetry, cd82 regions, sp80 trajectories); shape fitting (cn04, ka59, re86).
24. **Counting and arithmetic:** su15 (powers of two, counts), vc33 (heights), tr87 and sb26 (sequence lengths), ft09
    (colour index mod n).
25. **Action economy under a budget, lives or limited trials:** all 25; limited trials or lives in sp80 (5 pours),
    r11l (5 hazard hits), ls20 (3 lives), g50t (loop slots).

### Player / avatar
- Single avatar: tu93, wa30, ls20, g50t, dc22, sc25, bp35 (7).
- No avatar, but you steer selectable bodies or a tool: m0r0 (mirrored pair), ka59, sk48, re86, ar25, cn04, sp80,
  cd82 (bucket).
- Pure board/puzzle interfaces (no controlled body): ft09, lp85, lf52, s5i5, vc33, r11l, su15, tn36, tr87, sb26.

### Goal shown on screen
- Exact target state to copy: cd82 (picture), re86 (outline drawing), sk48 (reference arms), sb26 (colour sequence),
  su15 (item icons + goal zones), tn36 (target pose/colour outline), ls20 (required key on each door), ka59 (target
  frames of matching size).
- Target locations or markers: tu93, g50t, dc22, sc25, bp35 (exits/gems); wa30 (zone); lp85, s5i5, r11l, ar25, vc33
  (markers); sp80 (cups).
- Rule shown, answer must be derived: tr87 (rewrite rules), ft09 (constraint keys).
- Not shown at all: m0r0, cn04, lf52.

### Step / energy counters
- All 25 show one. Positions: row 63 (bp35, cd82, dc22, ft09, g50t timer sprite, ka59, re86, s5i5, su15, tr87, tu93,
  wa30); row 0 (cn04 x=16..47, lf52, sp80, vc33; m0r0 uses rows 0 and 63); row 53 (sk48, sb26; also divides the
  screen); column 0 (r11l, lp85); column 63 (ar25, multi-colour layers); right edge x~62 (sc25); rows 61-62 pips plus
  lives (ls20); top-left sliding bar (tn36).
- Counters that are not shown: failed-pour count in sp80, hazard-hit count in r11l, escalating error penalty in su15,
  the 20-unit fall penalty in dc22.
- What costs: most games charge every action; some charge only effective actions (ft09 tile clicks, lp85 button clicks,
  sk48 moves, ar25 moves and ACTION5). Undo is charged in bp35, free in ar25 and sk48, and in lf52 it moves the
  displayed bar but not the real limit.

### Irreversible / dead-end states
- Explicitly signalled dead ends: lf52 (pegs grey + restart button), su15 (goal zone greys).
- Irreversible without undo: m0r0 (merged pair), ka59 (objects kicked into corners, one-shot bombs), tu93 (death),
  sc25 (destroyed switch), ls20 (consumed doors/pickups; a life lost), g50t (crushed avatar), sp80 (each failed pour
  uses one of 5 trials), wa30 (removed thief), re86 (recolouring, likely).
- Irreversible but undo available: lf52, su15, bp35 (destroyed blocks), sb26, sk48, ar25.
- Fully reversible moves (dead ends only via budget): ft09, lp85, tr87, cn04, cd82, s5i5, r11l, dc22, tn36; vc33
  likely (unclear from source whether every pump has a reverse button).

### Frame-format surprises
- Multi-frame responses are the norm: tu93 (~12 frames per key), g50t (6-frame slides), sk48 (2 frames, 35 on win),
  sb26 and tn36 (program execution), su15 (ring, 16-frame error flash), sp80 (pour simulation), ka59 (slides,
  explosions), lf52 and bp35 (custom renderers that emit whole animation sequences per action), cd82 (14-frame paint).
  An agent must read the LAST frame as the state and may use earlier frames as evidence of dynamics.
- Whole-frame transforms: sp80 rotates the entire frame by 180 deg in L2, L3, L5 (HUD included); ls20 L7 blacks out
  everything beyond ~20 px of the avatar (fog); dc22 closes the screen to black in a shrinking circle on a fall; lf52
  dissolves the screen on restart; m0r0 repaints background pixels with two per-level colours.
- Upscaled small grids and letterboxes: m0r0 (11-15 cells), cn04 (20x20), ft09 (32x32), sp80 (16/20), ar25 (21x21),
  vc33 (32, 48 or 52 in some levels), tu93 (39-51), ka59 (45-63), lp85 (various), dc22 (64x44/48). Cell size in output pixels varies per level.
- Scrolling views (state off-screen): lf52, bp35.
- HUD sprites that look like game objects: ls20 key display, g50t loop slots, su15 goal icons, lp85 level pips, tn36
  timer bar, g50t timer bar (a game sprite that moves one pixel per two actions).
- Tutorial hints inside the frame: ar25 L2, ft09 L1, lf52 L1, ls20 L1, sc25 L1-3 (spell demos), su15 L1.
- RESET at the very start of a level restarts the whole game (see engine facts), which costs all progress.
