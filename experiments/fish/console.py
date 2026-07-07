"""Terminal command console for the running simulation.

While the pygame window runs, the launching terminal doubles as a control
console: ``/help`` lists everything, ``/set`` browses and live-edits config
knobs, ``/stats`` reports the ecosystem, and action commands mirror the
keyboard shortcuts. Output is rendered with ``rich``.

Threading contract: :func:`start_reader` runs a daemon thread that ONLY reads
stdin lines into a queue — it never touches the sim. The pygame main loop
drains the queue once per frame and calls :func:`execute` on the main thread,
so every sim/config mutation stays single-threaded with ``sim.step``.

Commands act through a small "world" adapter object (see ``LoopWorld`` in
``main.py``) with: ``cfg``, ``sim`` (assignable), ``paused``, ``manual``,
``set_gene(name)``, ``reset()`` and ``quit()``. Tests drive :func:`execute`
directly with such an adapter — no thread, no pygame, no rendering needed.
"""

from __future__ import annotations

import difflib
import queue
import threading
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from genome import SHARK_GENE_NAMES

out = Console(highlight=False)  # swappable in tests (record/StringIO consoles)

PROMPT = "sim> "


# --- Parameter registry -------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One runtime-settable ``Config`` field.

    ``kind`` is one of int / float / bool / choice / colors / ints; ``lo``/``hi``
    clamp numeric values (and each element of ``ints``). ``apply`` names a hook
    in ``APPLY_HOOKS`` run after the value lands (e.g. growing a population
    buffer); ``needs_reset`` marks values only read at ``Simulation()`` init.
    """

    kind: str
    lo: float | None = None
    hi: float | None = None
    category: str = "misc"
    help: str = ""
    choices: tuple[str, ...] = ()
    apply: str | None = None
    needs_reset: bool = False
    note: str = ""


# Post-set hooks, run with the world adapter. Growing a buffer is what makes a
# raised population cap usable immediately — appends assert against capacity.
APPLY_HOOKS = {
    "grow_fish": lambda w: w.sim.fish.ensure_capacity(max(w.cfg.num_fish, 8)),
    "grow_sharks": lambda w: w.sim.sharks.ensure_capacity(
        max(w.cfg.max_sharks, w.cfg.num_sharks)
    ),
    "rebuild_walls": lambda w: w.sim.rebuild_walls(),
    "sync_gene": lambda w: w.set_gene(w.cfg.shark_color_gene),
}

WALLS_NOTE = "applies on next rebuild (/walls rebuild)"

PARAMS: dict[str, Param] = {
    # --- speed ---
    "tps": Param("int", 10, 240, "speed", "sim ticks per second (world speed)"),
    "fps": Param("int", 10, 240, "speed", "render frames per second (smoothness)"),
    # --- world ---
    "wrap_edges": Param("bool", category="world", help="wrap-around vs soft-bounce edges"),
    "num_fish": Param(
        "int", 0, 4000, "world", "fish population cap (respawn refills to this)",
        apply="grow_fish",
    ),
    "num_sharks": Param(
        "int", 0, 500, "world", "starting shark population", needs_reset=True
    ),
    "max_sharks": Param(
        "int", 1, 500, "world", "hard cap on living sharks", apply="grow_sharks"
    ),
    "shark_cull_count": Param(
        "int", 0, 500, "world", "random sharks removed when the cap is hit (0=off)"
    ),
    # --- fish ---
    "fish_perception": Param("float", 5, 300, "fish", "neighbour visibility radius"),
    "fish_max_speed": Param("float", 0.5, 20, "fish", "fish top speed"),
    "separation_radius": Param("float", 0, 200, "fish", "personal-space distance"),
    "min_speed": Param("float", 0, 5, "fish", "drift floor so fish never freeze"),
    "max_force": Param("float", 0.01, 2, "fish", "steering clamp (Reynolds)"),
    "fish_w_sep": Param("float", 0, 10, "fish", "separation weight"),
    "fish_w_ali": Param("float", 0, 10, "fish", "alignment weight"),
    "fish_w_coh": Param("float", 0, 10, "fish", "cohesion weight"),
    "fish_w_flee": Param("float", 0, 10, "fish", "flee-from-shark weight"),
    "fish_respawn_rate": Param("float", 0, 50, "fish", "new fish per tick (fractional)"),
    "fish_clone_radius": Param("float", 2, 200, "fish", "a newborn appears within this of its parent"),
    "fish_clone_min_pop": Param("int", 0, 1000, "fish", "below this population, respawns scatter randomly"),
    "fish_colors": Param(
        "colors", category="fish", help="base palette, hex list (new spawns only)"
    ),
    "fish_color_variation": Param(
        "float", 0, 5, "fish", "colour spread around the palette (0=exact)"
    ),
    # --- predation ---
    "flee_radius": Param("float", 0, 500, "predation", "fish start fleeing a shark at this range"),
    "kill_radius": Param("float", 1, 100, "predation", "fish within this of a shark get eaten"),
    # --- shark GA ---
    "shark_start_energy": Param("float", 0, 10000, "shark", "energy of a fresh shark"),
    "shark_energy_per_fish": Param("float", 0, 10000, "shark", "energy gained per fish eaten"),
    "shark_repro_energy": Param("float", 0, 10000, "shark", "energy needed to spawn a child"),
    "shark_birth_cost": Param("float", 0, 5000, "shark", "energy lost from the split at birth"),
    "shark_metab_base": Param("float", 0, 10, "shark", "baseline energy burned per tick"),
    "shark_metab_speed_k": Param("float", 0, 1, "shark", "extra burn per speed² per tick"),
    "shark_metab_perc_k": Param("float", 0, 0.1, "shark", "extra burn per unit of perception"),
    "shark_mutation_rate": Param("float", 0, 1, "shark", "per-gene mutation chance at breeding"),
    "shark_mutation_sigma": Param("float", 0, 2, "shark", "gene nudge size (fraction of span)"),
    "shark_color_gene": Param(
        "choice", category="shark", help="which gene the shark colour maps to",
        choices=tuple(SHARK_GENE_NAMES), apply="sync_gene",
    ),
    "shark_max_turn": Param("float", 0.01, 3.14, "shark", "max heading change per tick (radians)"),
    "retina_count_norm": Param("float", 1, 100, "shark", "fish per retina sector reading as full"),
    "brain_init_scale": Param("float", 0, 5, "shark", "founder brain weight scale"),
    "brain_weight_cap": Param("float", 0.1, 10, "shark", "hard clip on every brain weight"),
    "brain_mutation_rate": Param("float", 0, 1, "shark", "per-weight nudge chance at breeding"),
    "brain_mutation_sigma": Param("float", 0, 2, "shark", "brain nudge size (absolute)"),
    "shark_memory_frames": Param(
        "int", 0, 8, "shark", "past sense frames fed to the brain", needs_reset=True
    ),
    "shark_hidden": Param(
        "ints", 1, 128, "shark", "hidden layer sizes of the shark MLP", needs_reset=True
    ),
    # --- rescue ---
    "rescue_count": Param("int", 0, 500, "rescue", "sharks spawned by a manual /rescue"),
    "min_sharks": Param("int", 0, 500, "rescue", "auto-repopulate to this on extinction"),
    "hof_size": Param("int", 1, 200, "rescue", "elite snapshots kept in the hall of fame"),
    "hof_age_weight": Param("float", 0, 10, "rescue", "age's weight in the elite score"),
    "rescue_mut_scale": Param("float", 0, 5, "rescue", "mutation multiplier on rescue clones"),
    # --- edges & walls ---
    "edge_margin": Param("float", 0, 300, "walls", "bounce mode: start turning inward here"),
    "edge_turn": Param("float", 0, 5, "walls", "bounce mode: inward nudge per tick"),
    "walls_enabled": Param(
        "bool", category="walls", help="scatter wall shards across the sea",
        apply="rebuild_walls",
    ),
    "num_walls": Param("int", 0, 200, "walls", "how many shards", note=WALLS_NOTE),
    "wall_min_radius": Param("float", 5, 200, "walls", "smallest shard size", note=WALLS_NOTE),
    "wall_max_radius": Param("float", 5, 200, "walls", "largest shard size", note=WALLS_NOTE),
    "wall_margin": Param("float", 5, 100, "walls", "soft-bounce band around shards"),
    "wall_push": Param("float", 0, 5, "walls", "velocity nudge near a wall face"),
    "wall_border_margin": Param("float", 0, 300, "walls", "keep shards off the screen edges", note=WALLS_NOTE),
}

CATEGORY_ORDER = ("speed", "world", "fish", "predation", "shark", "rescue", "walls")


# --- Value parsing / formatting ------------------------------------------------


def _parse_bool(text: str) -> bool:
    t = text.lower()
    if t in ("on", "true", "yes", "1"):
        return True
    if t in ("off", "false", "no", "0"):
        return False
    raise ValueError(f"expected on/off, got {text!r}")


def _parse_color(text: str) -> tuple[int, int, int]:
    t = text.strip().lstrip("#")
    if len(t) != 6:
        raise ValueError(f"expected hex colour like #ff785a, got {text!r}")
    return tuple(int(t[i : i + 2], 16) for i in (0, 2, 4))


def parse_value(param: Param, text: str):
    """Parse ``text`` per the param's kind; clamp numerics into [lo, hi].

    Returns ``(value, clamped)`` where ``clamped`` says the input was pulled
    back into range (worth telling the user about)."""
    if param.kind == "bool":
        return _parse_bool(text), False
    if param.kind == "choice":
        if text not in param.choices:
            raise ValueError(f"expected one of {', '.join(param.choices)}")
        return text, False
    if param.kind == "colors":
        colors = tuple(_parse_color(c) for c in text.split(",") if c.strip())
        if not colors:
            raise ValueError("expected at least one hex colour")
        return colors, False
    if param.kind == "ints":
        values = tuple(int(v) for v in text.replace(" ", "").split(",") if v)
        if not (1 <= len(values) <= 4):
            raise ValueError("expected 1-4 comma-separated layer sizes")
        clamped_values = tuple(int(min(max(v, param.lo), param.hi)) for v in values)
        return clamped_values, clamped_values != values
    value = float(text)
    clamped = min(max(value, param.lo), param.hi)
    if param.kind == "int":
        clamped = int(round(clamped))
    return clamped, clamped != value


def format_value(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple):  # colour palette
            return ",".join("#" + "".join(f"{c:02x}" for c in rgb) for rgb in value)
        return ",".join(str(v) for v in value)
    return str(value)


# --- Command handlers -----------------------------------------------------------

COMMANDS: dict[str, tuple] = {}  # name -> (handler, usage, description)


def command(name: str, usage: str, desc: str):
    def register(fn):
        COMMANDS[name] = (fn, usage, desc)
        return fn

    return register


def _error(msg: str) -> None:
    out.print(f"[red]{msg}[/red]")


def _suggest(name: str, pool) -> str:
    close = difflib.get_close_matches(name, pool, n=1)
    return f" — did you mean [bold]{close[0]}[/bold]?" if close else ""


def _params_table(names, cfg, title: str) -> Table:
    table = Table(title=title, header_style="bold cyan", title_justify="left")
    for col in ("param", "value", "range", "info"):
        table.add_column(col)
    ordered = sorted(names, key=lambda n: (CATEGORY_ORDER.index(PARAMS[n].category), n))
    last_cat = None
    for name in ordered:
        p = PARAMS[name]
        if p.category != last_cat:
            table.add_section()
            last_cat = p.category
        rng = f"{format_value(p.lo)}..{format_value(p.hi)}" if p.lo is not None else p.kind
        notes = p.help + (" [yellow](needs /reset)[/yellow]" if p.needs_reset else "")
        if p.note:
            notes += f" [yellow]({p.note})[/yellow]"
        table.add_row(
            f"[bold]{name}[/bold]", format_value(getattr(cfg, name)), rng, notes
        )
    return table


def set_param(name: str, text: str, world) -> None:
    if name not in PARAMS:
        _error(f"unknown parameter {name!r}{_suggest(name, PARAMS)}")
        return
    param = PARAMS[name]
    try:
        value, clamped = parse_value(param, text)
    except ValueError as err:
        _error(f"{name}: {err}")
        return
    old = getattr(world.cfg, name)
    setattr(world.cfg, name, value)
    if param.apply:
        APPLY_HOOKS[param.apply](world)
    msg = f"[green]{name}: {format_value(old)} → {format_value(value)}[/green]"
    if clamped:
        msg += f" [yellow](clamped to {format_value(param.lo)}..{format_value(param.hi)})[/yellow]"
    if param.needs_reset:
        msg += " [yellow]— takes effect on /reset[/yellow]"
    if param.note:
        msg += f" [yellow]— {param.note}[/yellow]"
    out.print(msg)


@command("help", "/help", "list all commands and keyboard shortcuts")
def cmd_help(args, world) -> None:
    table = Table(title="commands", header_style="bold cyan", title_justify="left")
    table.add_column("command")
    table.add_column("what it does")
    for name, (_fn, usage, desc) in COMMANDS.items():
        table.add_row(f"[bold]{usage}[/bold]", desc)
    out.print(table)
    out.print(
        "keyboard: [bold]T[/bold] tps  [bold]F[/bold] fps  [bold]W[/bold] edges  "
        "[bold]O[/bold] walls  [bold]M[/bold] mouse-shark  [bold]K[/bold] shark colour  "
        "[bold]B[/bold] rescue  [bold]R[/bold] reset  [bold]Space[/bold] pause  "
        "[bold]Esc[/bold] quit"
    )


@command("set", "/set [name] [value]", "browse (no args / prefix) or change a parameter")
def cmd_set(args, world) -> None:
    if not args:
        out.print(_params_table(PARAMS, world.cfg, "all parameters"))
        return
    if len(args) == 1:
        name = args[0]
        if name in PARAMS:
            out.print(_params_table([name], world.cfg, name))
            return
        matches = [n for n in PARAMS if name.lower() in n]
        if matches:
            out.print(_params_table(matches, world.cfg, f"parameters matching {name!r}"))
        else:
            _error(f"nothing matches {name!r}{_suggest(name, PARAMS)}")
        return
    set_param(args[0], " ".join(args[1:]), world)


@command("get", "/get <name>", "show one parameter's current value")
def cmd_get(args, world) -> None:
    if len(args) != 1:
        _error("usage: /get <name>")
        return
    if args[0] not in PARAMS:
        _error(f"unknown parameter {args[0]!r}{_suggest(args[0], PARAMS)}")
        return
    out.print(f"{args[0]} = [bold]{format_value(getattr(world.cfg, args[0]))}[/bold]")


@command("config", "/config", "dump the full config, including read-only values")
def cmd_config(args, world) -> None:
    out.print(_params_table(PARAMS, world.cfg, "all parameters"))
    out.print(
        f"read-only: width={world.cfg.width} height={world.cfg.height} "
        "(resize the window)"
    )


@command("stats", "/stats", "live ecosystem statistics")
def cmd_stats(args, world) -> None:
    s = world.sim.stats
    table = Table(title="ecosystem", header_style="bold cyan", title_justify="left")
    for col in ("", "population", "born", "died", "extra"):
        table.add_column(col)
    table.add_row(
        "fish",
        f"{int(s['population'])}/{world.cfg.num_fish}",
        str(int(s["births"])),
        str(int(s["deaths"])),
        "",
    )
    genes = "  ".join(f"{n}:{s['shark_' + n]:.1f}" for n in SHARK_GENE_NAMES)
    table.add_row(
        "sharks",
        f"{int(s['shark_population'])}/{world.cfg.max_sharks}",
        str(int(s["shark_births"])),
        str(int(s["shark_deaths"])),
        f"gen:{s['shark_generation']:.1f} energy:{s['shark_energy']:.1f} "
        f"hof:{len(world.sim.hall_of_fame)}  {genes}",
    )
    out.print(table)
    out.print(
        f"paused: {format_value(world.paused)}   manual shark: {format_value(world.manual)}   "
        f"edges: {'wrap' if world.cfg.wrap_edges else 'bounce'}   "
        f"walls: {format_value(world.cfg.walls_enabled)}"
    )


@command("pause", "/pause", "pause the simulation")
def cmd_pause(args, world) -> None:
    world.paused = True
    out.print("[green]paused[/green]")


@command("resume", "/resume", "resume the simulation")
def cmd_resume(args, world) -> None:
    world.paused = False
    out.print("[green]resumed[/green]")


@command("reset", "/reset", "rebuild the world from the current config")
def cmd_reset(args, world) -> None:
    world.reset()
    out.print("[green]world reset[/green]")


@command("rescue", "/rescue [n]", "breed sharks back from the hall of fame")
def cmd_rescue(args, world) -> None:
    try:
        n = int(args[0]) if args else world.cfg.rescue_count
    except ValueError:
        _error("usage: /rescue [count]")
        return
    before = len(world.sim.sharks)
    world.sim.rescue_sharks(n)
    out.print(f"[green]rescued {len(world.sim.sharks) - before} sharks[/green]")


@command("walls", "/walls on|off|rebuild", "toggle wall shards or deal a fresh layout")
def cmd_walls(args, world) -> None:
    mode = args[0].lower() if args else None
    if mode not in ("on", "off", "rebuild", None):
        _error("usage: /walls on|off|rebuild")
        return
    if mode is None:
        out.print(f"walls: {format_value(world.cfg.walls_enabled)}")
        return
    if mode in ("on", "off"):
        world.cfg.walls_enabled = mode == "on"
    world.sim.rebuild_walls()
    out.print(f"[green]walls {mode}: {len(world.sim.walls.shards)} shards[/green]")


@command("edges", "/edges wrap|bounce", "world edge behaviour")
def cmd_edges(args, world) -> None:
    mode = args[0].lower() if args else None
    if mode not in ("wrap", "bounce", None):
        _error("usage: /edges wrap|bounce")
        return
    if mode is not None:
        world.cfg.wrap_edges = mode == "wrap"
    out.print(f"edges: [bold]{'wrap' if world.cfg.wrap_edges else 'bounce'}[/bold]")


@command("manual", "/manual", "toggle the mouse-driven shark")
def cmd_manual(args, world) -> None:
    world.manual = not world.manual
    out.print(f"manual shark: [bold]{format_value(world.manual)}[/bold]")


@command("gene", "/gene <name>", "which shark gene the colour shows")
def cmd_gene(args, world) -> None:
    if len(args) != 1 or args[0] not in SHARK_GENE_NAMES:
        _error(f"usage: /gene {{{'|'.join(SHARK_GENE_NAMES)}}}")
        return
    world.cfg.shark_color_gene = args[0]
    world.set_gene(args[0])
    out.print(f"shark colour gene: [bold]{args[0]}[/bold]")


@command("spawn", "/spawn fish|shark <n>", "inject animals at random spots (capped)")
def cmd_spawn(args, world) -> None:
    if len(args) != 2 or args[0] not in ("fish", "shark"):
        _error("usage: /spawn fish|shark <count>")
        return
    try:
        n = max(0, int(args[1]))
    except ValueError:
        _error("usage: /spawn fish|shark <count>")
        return
    sim, cfg = world.sim, world.cfg
    if args[0] == "fish":
        k = max(0, min(n, cfg.num_fish - len(sim.fish)))
        for _ in range(k):
            sim.add_fish(sim.rng.uniform(0, sim.world))
    else:
        k = max(0, min(n, cfg.max_sharks - len(sim.sharks)))
        for _ in range(k):
            sim.add_shark(sim.rng.uniform(0, sim.world))
    capped = " [yellow](population cap)[/yellow]" if k < n else ""
    out.print(f"[green]spawned {k} {args[0]}[/green]{capped}")


@command("tps", "/tps <n>", "set the sim tick rate (world speed)")
def cmd_tps(args, world) -> None:
    set_param("tps", args[0], world) if args else cmd_get(["tps"], world)


@command("fps", "/fps <n>", "set the render frame rate (smoothness)")
def cmd_fps(args, world) -> None:
    set_param("fps", args[0], world) if args else cmd_get(["fps"], world)


@command("quit", "/quit", "close the simulation")
def cmd_quit(args, world) -> None:
    out.print("[green]bye[/green]")
    world.quit()


# --- Entry points ---------------------------------------------------------------


def execute(line: str, world) -> None:
    """Parse and run one console line against the world. Never raises: a
    console mistake (or bug) must not take the render loop down with it."""
    line = line.strip()
    if not line:
        return
    if not line.startswith("/"):
        _error("commands start with / — try /help")
        return
    name, *args = line[1:].split()
    name = name.lower()
    if name not in COMMANDS:
        _error(f"unknown command /{name}{_suggest(name, COMMANDS)} — try /help")
        return
    try:
        COMMANDS[name][0](args, world)
    except Exception as err:  # noqa: BLE001 — surface, don't crash the sim
        _error(f"/{name} failed: {err}")


def start_reader(commands: queue.Queue) -> threading.Thread:
    """Read stdin lines into ``commands`` from a daemon thread (EOF → /quit)."""

    def read_loop() -> None:
        while True:
            try:
                commands.put(input(PROMPT))
            except (EOFError, KeyboardInterrupt):
                commands.put("/quit")
                return

    thread = threading.Thread(target=read_loop, daemon=True, name="console-reader")
    thread.start()
    return thread
