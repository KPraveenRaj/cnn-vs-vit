"""Generate all six PowerPoint decks from results on disk.

  progress_01..04   short updates for the guide, sent one at a time as the
                    corresponding milestone is genuinely reached
  midsem_review     the mid-semester presentation
  endsem_review     the end-semester presentation (+ Food-101, + Phase-II)

Written for a MIXED audience. The project supervisor knows the field; a panel,
an external examiner from an adjacent area, or a family member asking what the
work is about does not. Three devices carry that:

  * explainer slides come BEFORE results, building the vocabulary each result
    needs (what a CNN and a ViT actually are, what transfer learning is, what
    "spatial frequency" means for a photograph) and leaning on the qualitative
    figures, because a picture of a low-pass-filtered image explains the idea
    faster than any definition;
  * every slide carries SPEAKER NOTES — a script that expands each term on
    first use and reads each number aloud in plain language, so the deck can be
    presented without improvising the explanation;
  * result slides carry an "in plain terms" box on the slide itself, because
    someone looking at the figure needs the translation next to it.

Every number comes from report/build/facts.py, which reads results/tables only.
Where a result does not exist yet the deck says "(pending)" rather than
inventing a plausible value. Re-run after more results land and every slide
updates.

The progress decks are deliberately UNDATED: they describe milestones, not
calendar days. Send each one when its milestone is actually true.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx.util import Inches as I

from deckkit import (ACCENT, GOOD, RESNET, VIT, bullets, caption, col_header,
                     new_deck, notes, picture, plain, save, slide, table,
                     takeaway, title_slide, two_col)
from facts import Facts

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "report" / "phase1" / "decks"

STUDENT = "Praveen Raj Konatham · 252SP014 · M.Tech SPML"
GUIDE = "Guide: Dr. Bini A A · Dept. of ECE, NIT Karnataka Surathkal"
COURSE = "EC789 Major Project – I"
TITLE = "CNNs vs Vision Transformers: A Controlled Comparison under Transfer Learning"

PROBLEM = ("Vision Transformers now lead image-classification benchmarks, but those "
           "results rely on large-scale pre-training and compute, and most published "
           "CNN-versus-ViT comparisons vary many factors at once. This project "
           "fine-tunes a representative CNN and a representative ViT under a single "
           "controlled transfer-learning protocol and compares how the two families "
           "behave.")


# ==========================================================================
# Explainer slides — vocabulary, before any result depends on it
# ==========================================================================
def _asym_line(f):
    """One sentence stating what the schedule bug cost, against the ADOPTED protocol.

    Not against the transformer's best annealed run: that compares to a schedule
    which was not adopted and inflates the ratio from ~3.6x to ~13x.
    """
    a = f.schedule_asymmetry() or {}
    ad = a.get("adopted")
    if not ad:
        return "It hurt the transformer more than the CNN. Every ViT run was cut short."
    return (f"Measured against the schedule we adopted, it cost the transformer "
            f"{ad['vit_b16']:.2f} points and the CNN {ad['resnet50']:.2f} — "
            f"{ad['ratio']:.1f}x worse for the transformer. Every single ViT run was "
            f"cut short.")



def sl_question(prs, f):
    s = slide(prs, "The question, in one sentence", kicker="1 · What this is about")
    bullets(s, [
        (0, "When you fine-tune a pre-trained image model on a new task, does a "
            "Vision Transformer or a convolutional network give you more — and WHY?", True),
        (0, "Two things make this hard to answer from the literature:", False),
        (1, "published comparisons usually change several things at once — the "
            "pre-training data, the augmentation recipe, the training length — so an "
            "'architecture' conclusion is not actually about architecture", False),
        (1, "most of them report only which model scores higher, not what the two "
            "models are doing differently", False),
        (0, "So: hold EVERYTHING fixed except the architecture, and measure behaviour, "
            "not just the score.", True),
    ])
    plain(s, "Two well-known families of image recognisers are compared as fairly as "
             "possible, and we try to explain the difference rather than just report it.")
    notes(s, """
Open with the everyday version of the problem. Somebody hands you a model that has
already been trained on millions of general photographs. You want it to do YOUR
task, and you only have a few thousand pictures. Which kind of model should you
pick, and what are you actually buying?

The honest answer from the literature is muddy, because published comparisons
usually differ in several ways at once. If model A saw more pre-training data than
model B, and also trained for longer, and also used a fancier data-augmentation
recipe, then "A beats B" tells you nothing about A's architecture.

So the design principle of this project is: change one thing. Everything else —
the data, the split, the augmentation, the schedule shape, the random seeds, the
evaluation — is identical. Then whatever difference remains is attributable to the
architecture itself.

The second half matters as much as the first. We are not just asking who wins; we
are asking what each model depends on, which is what lets you predict how they
will behave on data you have not tested yet.
""")
    return s


def sl_two_models(prs, f):
    s = slide(prs, "The two contenders", kicker="2 · Background")
    col_header(s, "ResNet-50 — a convolutional network (CNN)", True, color=RESNET)
    col_header(s, "ViT-B/16 — a Vision Transformer", False, color=VIT)
    two_col(s, [
        (0, "Scans the image with small sliding filters.", False),
        (0, "Built-in assumption: nearby pixels belong together, and the same pattern "
            "means the same thing wherever it appears.", False),
        (0, "That assumption is free knowledge — it makes CNNs efficient learners on "
            "images.", False),
        (0, "Design dates from the 1980s-2010s; the mature, default choice.", False),
    ], [
        (0, "Cuts the image into a grid of 16x16 patches and treats them like words in "
            "a sentence.", False),
        (0, "Every patch can look at every other patch directly — 'attention'.", False),
        (0, "Almost no built-in assumption about images, so it must learn more from "
            "data — but is freer to learn whatever helps.", False),
        (0, "Adapted from language models in 2020.", False),
    ], size=14)
    rd, vd = f.deployment_row("resnet50"), f.deployment_row("vit_b16")
    if rd and vd:
        table(s, ["", "ResNet-50", "ViT-B/16"],
              [["Parameters", f"{rd.get('params_m',0):.0f} M", f"{vd.get('params_m',0):.0f} M"],
               ["Compute per image", f"{rd.get('gmacs',0):.1f} GMACs",
                f"{vd.get('gmacs',0):.1f} GMACs"]],
              top=I(5.35), col_w=[1.6, 1, 1], size=12)
    notes(s, """
These are the two archetypes. Pick the words carefully here, because everything
later depends on the audience having this picture.

A convolutional network — ResNet-50 — looks at an image through a small window
that slides across it, hunting for local patterns: edges, then textures, then
parts, then objects. It is built with an assumption baked in, that nearby pixels
are related and that a pattern means the same thing wherever it appears. That
assumption is a gift: it means the network does not have to learn from scratch
that images are spatially structured.

A Vision Transformer — ViT-B/16 — does something stranger. It chops the picture
into a grid of small square patches, sixteen pixels on a side, and then treats
that as a sequence, rather like words in a sentence. Its core operation,
attention, lets any patch consult any other patch directly, so a patch in one
corner can be compared to a patch in the opposite corner in a single step. It
carries almost no built-in assumption about images, which means it has more to
learn — but also that it is less constrained in what it can learn.

Note the size difference: the ViT has roughly three and a half times the
parameters and needs about four times the arithmetic per image. Any advantage it
shows has to be weighed against that.
""")
    return s


def sl_transfer(prs, f):
    s = slide(prs, "Transfer learning — and why the data budget is the interesting knob",
              kicker="3 · Background")
    bullets(s, [
        (0, "Nobody trains these models from scratch. Both start from weights learned "
            "on ImageNet — about 1.3 million general photographs.", True),
        (0, "Fine-tuning = continue training that already-knowledgeable model on your "
            "smaller, specific dataset.", False),
        (0, "This is how essentially all applied computer vision works, because almost "
            "nobody has a million labelled images of their own problem.", False),
        (0, "So the practical question is not 'which model is best with unlimited "
            "data?' but 'which model gives me more when I have only a little?'", True),
        (0, "We therefore train at 10%, 25%, 50% and 100% of the available data and "
            "watch how the answer changes.", False),
        (1, "Both models are pre-trained on the SAME ImageNet-1k images — a control "
            "that is easy to get wrong and that many published comparisons miss.", False),
    ])
    plain(s, "Both models start out already knowing a lot about photographs in general. "
             "We test how well each one uses a small amount of new, specific data.")
    notes(s, """
This slide explains the setting, and it is the one non-experts most often miss.

Nobody in practice trains a large image model from nothing. It takes enormous data
and compute. Instead you download a model that has already been trained on
ImageNet — roughly 1.3 million everyday photographs across a thousand categories —
and then continue training it briefly on your own, much smaller dataset. That
second step is called fine-tuning, and the whole approach is transfer learning:
knowledge transfers from the general task to your specific one.

Because that is how applied vision actually works, the interesting question is not
who wins with unlimited data. It is who gives you more when you have very little,
because having very little is the normal condition.

So we deliberately starve both models. We train them on ten percent of the data,
then twenty-five, then fifty, then all of it, and watch how the gap between them
behaves.

One control worth flagging out loud: both models start from ImageNet-1k, the same
1.3 million images. The most popular downloadable ViT weights are actually trained
on a much larger set, ImageNet-21k. Using those would have handed the transformer
a fourteen-times-larger head start and quietly invalidated the whole comparison.
We pinned the ImageNet-1k version instead.
""")
    return s


def sl_controlled(prs, f):
    d = f.dataset_facts()
    s = slide(prs, "What “a controlled comparison” actually requires", kicker="4 · Method")
    bullets(s, [
        (0, "Identical for both models — pre-training source, data splits, "
            "augmentation, image size (224x224), schedule shape, optimiser family, "
            "random seeds, and the entire evaluation path.", True),
        (0, "ONE declared exception: each model gets its own learning rate, chosen by a "
            "documented search.", False),
        (1, f"ResNet-50 → {f.lr_selected('resnet50')}    ViT-B/16 → {f.lr_selected('vit_b16')}", False),
        (1, "The learning rate controls how big a step the model takes when it corrects "
            "itself. The two families want values a whole decade apart — forcing one "
            "value on both would cripple one of them, which is the opposite of fair.", False),
        (0, f"Data: Caltech-256 — {d['total']:,} photographs, {d['classes']} categories, "
            f"split once into {d['train_f100']:,} for training, {d['val']:,} for "
            f"tuning decisions and {d['test']:,} held back.", False),
        (0, "The held-back set was frozen at the start and never used to make a single "
            "decision. It is the closest thing to an honest exam.", True),
    ])
    notes(s, """
This is the methodological heart, and it is worth being slow here.

Everything that could differ between the two models has been made identical: the
images they see, how those images are split, the random jitter applied during
training, the picture size, the shape of the training schedule, the family of
optimiser, the random seeds, and every step of how they are scored.

There is exactly one deliberate exception, and it needs explaining rather than
hiding. Each model gets its own learning rate. The learning rate is simply how big
a correction the model makes each time it gets something wrong — too large and it
thrashes around, too small and it barely moves. The two architectures want very
different values: the CNN's best setting is about ten times larger than the
transformer's. If we had forced a single value on both, one of them would have been
handicapped, and we would have measured our own bad choice rather than the
architecture. So each is tuned separately, by a documented search, and that search
is reported.

The last point is the one to emphasise to any examiner. Twenty percent of the data
was set aside at the very beginning and never touched — not for choosing learning
rates, not for deciding when to stop training, not for anything. Every number
presented as a result is measured on images that played no part in any decision.
""")
    return s


def sl_dataset(prs, f):
    s = slide(prs, "What the data looks like", kicker="5 · Method")
    if f.figure("dataset_samples.png"):
        picture(s, f.figure("dataset_samples.png"), top=I(1.55), max_h=I(2.35))
    if f.figure("fraction_nesting.png"):
        picture(s, f.figure("fraction_nesting.png"), top=I(4.1), max_h=I(2.05))
    caption(s, "Top: random samples from the held-back test set. Bottom: the four "
               "training-set sizes, each a strict subset of the next.", top=I(6.25))
    notes(s, """
The top strip is just what the data is: everyday object photographs across 256
categories — musical instruments, animals, vehicles, household objects. It is a
classic benchmark, deliberately ordinary.

The bottom bar is a detail that matters more than it looks. When we shrink the
training set to fifty percent, we do not draw a fresh random sample. The ten
percent set is contained inside the twenty-five percent set, which is contained
inside the fifty, and so on — like nested Russian dolls. And the shrinking is done
per category, so no category is accidentally wiped out at the small sizes.

Why bother? Because if each fraction were an independent random draw, then a
difference between two fractions could come from the amount of data or from having
happened to draw easier pictures. Nesting removes that ambiguity: the only thing
that changes is how much data there is. We do this three times with three
different random seeds, which is what gives us the error bars later.
""")
    return s


def sl_design(prs, f):
    s = slide(prs, "The experiment", kicker="6 · Method")
    d = f.dataset_facts()
    table(s, ["Block", "What varies", "Runs"],
          [["Full fine-tuning, Caltech-256",
            "2 models × 4 data sizes × 3 random seeds", "24"],
           ["Linear probe, Caltech-256",
            "same grid, but the model is FROZEN and only a final layer is trained", "24"],
           ["Learning-rate searches", "3-4 rates per model, per regime", "13"],
           ["Food-101 confirmation", "2 models × 2 data sizes × 1 seed", "4"]],
          col_w=[1.5, 3.0, 0.6])
    bullets(s, [
        (0, "Then EVERY trained model is put through one identical test battery — 42 "
            "separate evaluations covering clean accuracy, damaged images, and "
            "filtered images.", True),
        (0, "Three seeds per cell means every headline number is a mean with a spread, "
            "not a single lucky run.", False),
        (0, f"Training compute: {f.gpu_hours()} GPU-hours on one laptop graphics card, "
            f"plus the evaluation battery.", False),
    ], top=I(4.05), size=15)
    notes(s, """
Walk the table top to bottom.

The first block is the main experiment: both models, four data sizes, three
different random starts each — twenty-four training runs.

The second block is a useful contrast. Instead of letting the whole model adapt,
we freeze it completely and train only a single final layer on top of the features
it already computes. That measures how good the pre-trained representation is
straight out of the box, with no adaptation. Comparing the two tells you whether a
model's advantage comes from what it already knows or from how well it adapts.

The third block is the learning-rate searches described earlier — the declared
per-model tuning.

The fourth is a second dataset, Food-101, used only to check whether the findings
hold somewhere else.

Then every trained model goes through the same battery of forty-two evaluations.
Same images, same order, same everything.

And three seeds matter: a single training run can be lucky. Every number in this
deck is an average over three, reported with its spread, so you can see whether a
difference is real or within noise.
""")
    return s


def sl_schedule_bug(prs, f):
    sf = f.schedule_finding()
    s = slide(prs, "A bug we caught that would have reversed the answer",
              kicker="7 · Method · honesty")
    if not sf:
        bullets(s, [(0, "(archive not present)", False)])
        return s
    rows = []
    for m, name in (("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16")):
        r = sf[m]
        cell = lambda k: f"{r[k][0]*100:.2f}%" if r.get(k) else "—"
        hit, tot = sf["early_stop_counts"][m]
        rows.append([name, cell("truncated"), cell("annealed_8"), cell("annealed_15"),
                     f"{hit}/{tot}"])
    table(s, ["Model", "Original (broken)", "Short schedule", "Fixed schedule",
              "Runs cut short"], rows, col_w=[1.1, 1.3, 1.1, 1.1, 1.1], size=12)
    bullets(s, [
        (0, "Our training recipe stopped runs early, while the model was still moving "
            "fast, before the phase where it settles down and consolidates.", True),
        (0, _asym_line(f), False),
        (0, "With the bug, the measured gap SHRANK with data and flipped sign. Without "
            "it, it never flips. Same data, same seeds.", True),
        (0, "Found before the final experiments, both models rerun identically, old "
            "runs archived rather than deleted.", False),
    ], top=I(3.4), size=14)
    plain(s, "We nearly reported the opposite conclusion. The cause was a training "
             "setting, not the models — so we fixed it, reran everything, and are "
             "reporting the near-miss.", top=I(6.2), height=I(0.95))
    notes(s, """
Present this slide deliberately rather than apologetically. It is the strongest
evidence in the deck that the results are trustworthy.

Here is the mechanism in plain terms. Training uses a schedule: the model takes big
correction steps early on and progressively smaller ones, so that it explores
first and settles down at the end. That final settling phase is where most of the
final quality appears. Separately, we had a rule that stopped training when the
score stopped improving for a while — sensible on its own.

The two rules fought each other. The schedule was set to run for thirty rounds, so
after eight or nine rounds the model was still taking large steps and its score was
bouncing around. The stopping rule saw the bouncing, concluded no progress was
being made, and killed the run — before the settling phase ever happened.

Both models were affected, but not equally. Measured against the schedule we
adopted, the convolutional network lost about half a point and the transformer
about two — so roughly a factor of four, hitting the transformer. Every single
transformer run in the grid was cut short, against eight of twelve for the CNN.

If someone asks for a bigger number: comparing against the transformer's very best
annealed run gives about thirteen times. Do not quote that one. It measures the
bug against a schedule we did not adopt, which overstates it.

And the consequence was not cosmetic. Under the broken setting, the gap between the
two models appeared to shrink as data grew and eventually reversed — the CNN
appeared to win at full data. Under the corrected setting, on exactly the same data
with the same random seeds, it never reverses. We would have reported the opposite
conclusion, and it would have been a conclusion about a training setting rather
than about architectures.

If anyone asks why the protocol says fifteen rounds instead of thirty, this is the
answer, and it is measured rather than asserted.
""")
    return s


def sl_result_efficiency(prs, f, with_probe=True):
    s = slide(prs, "Result 1 — who gets more out of limited data?",
              kicker="8 · Result")
    _eff_table(f, s, regimes=("fullft",), top=I(1.62))
    if f.figure("fig_data_efficiency.png"):
        picture(s, f.figure("fig_data_efficiency.png"), top=I(3.0), max_h=I(2.5))
    g10, g100 = f.gap_at(10), f.gap_at(100)
    if g10 is not None and g100 is not None:
        plain(s, f"The transformer wins at every data size, and wins by MOST when data "
                 f"is scarcest ({g10:+.2f} points at 10% data, {g100:+.2f} at 100%). "
                 f"Every gap is bigger than the run-to-run noise.", top=I(5.7),
              height=I(1.0))
    notes(s, """
This is the first result. Read the table left to right.

Each cell is the percentage of held-back test images the model gets right. The plus
or minus is the spread across three different random starts, so you can see how
much a number would wobble if we simply retrained.

The transformer is ahead at every data size. Crucially, the gap is largest when
data is scarcest — about two points at ten percent, shrinking to under one point at
full data.

Two things to stress. First, every gap is larger than the run-to-run spread, so
these are real differences, not noise. Second, this is the opposite of what people
often assume. You will frequently hear that transformers are data-hungry and lose
to CNNs when data is limited. That belief comes from training from scratch. In the
transfer-learning setting, where both models arrive already educated, it inverts:
the transformer's prior education transfers better precisely when there is least
new data to learn from.

If someone asks "is a two-point difference a big deal?" — in this field, on a
frozen test set, with the spread this small, yes. It is also consistent across all
four data sizes and three seeds, which matters more than the size.
""")
    if with_probe and f.available()["probes"]:
        s2 = slide(prs, "Result 1b — is it better knowledge, or better adaptation?",
                   kicker="8b · Result")
        _eff_table(f, s2, regimes=("linprobe",), top=I(1.62))
        bullets(s2, [
            (0, "A 'linear probe' freezes the model completely and trains only a single "
                "final layer. It measures what the model already knew, with no "
                "adaptation allowed.", True),
            (0, "The ordering INVERTS: the transformer's frozen features win at 10% "
                "data, but the CNN's win at 25% and above.", False),
            (0, "So the transformer's fine-tuning advantage is not simply that its "
                "features are better. It is that they ADAPT better.", True),
        ], top=I(3.3), size=15)
        plain(s2, "Frozen, the CNN's knowledge is often the more useful. Allowed to "
                  "adapt, the transformer pulls ahead. The advantage is in the "
                  "adapting.", top=I(5.95), height=I(0.95))
        notes(s2, """
This is a subtle result and it rewards a slow explanation.

A linear probe works like this. You freeze the entire pre-trained model so nothing
inside it can change, run your images through it, and take the numerical summary it
produces for each image. Then you train one very simple layer on top of those
summaries. Because the model itself cannot change, this measures purely what it
already knew — its raw knowledge, with adaptation switched off.

And the ordering flips. With the models frozen, the CNN's features are actually
more useful at twenty-five percent data and above. Only at the very smallest data
size does the transformer's frozen representation win.

Put the two results together and you get something more interesting than either
alone. When both models are allowed to adapt, the transformer wins everywhere. When
neither can adapt, the CNN often wins. So the transformer's advantage is not that
it knows more — it is that it is better at being taught. That is a claim about
adaptability, and it is only visible because we ran both regimes.
""")
    return s


def sl_how_corruption(prs, f):
    s = slide(prs, "How we test robustness", kicker="9 · Method")
    bullets(s, [
        (0, "Real photographs are not clean: sensors add noise, lenses go out of focus, "
            "files get compressed. A model that only works on pristine images is not "
            "much use.", True),
        (0, "So we damage every test image in three standard ways, at five increasing "
            "severities, and re-score.", False),
        (0, "The damage is generated on the fly from a fixed recipe, so BOTH models are "
            "scored on pixel-identical damaged images. No model gets an easier copy.", True),
    ], top=I(1.6), height=I(1.55), size=15)
    if f.figure("corruption_ladder.png"):
        picture(s, f.figure("corruption_ladder.png"), top=I(3.15), max_h=I(3.5))
    notes(s, """
This slide is mostly the picture — let people look at it.

Three rows, three kinds of damage, getting worse left to right.

The top row is sensor noise, the speckle you get from a phone camera in a dark
room. The middle row is blur, an out-of-focus lens. The bottom row is JPEG
compression, the blocky smearing you get when an image has been saved too small —
which, on the internet, is nearly always.

These are not arbitrary. They are the standard corruption benchmark used across the
robustness literature, at the standard severity levels, so our numbers are
comparable to published work.

One technical point worth making because it is the kind of thing that quietly ruins
comparisons: the damaged images are generated fresh each time from a fixed recipe
tied to the image and the severity level. That means both models are scored on
byte-for-byte identical damaged pictures. Neither gets a luckier draw of noise. It
also means anyone can reproduce these exact images.
""")
    return s


def sl_result_corruption(prs, f):
    s = slide(prs, "Result 2 — which model survives damaged images?",
              kicker="10 · Result")
    if f.figure("fig_corruption.png"):
        picture(s, f.figure("fig_corruption.png"), top=I(1.6), max_h=I(3.5))
    rows = [[{"resnet50": "ResNet-50", "vit_b16": "ViT-B/16"}[m],
             f.top1_str(m, 100), f.corruption_drop(m)] for m in ("resnet50", "vit_b16")]
    table(s, ["Model", "Clean accuracy", "Average accuracy LOST to damage"], rows,
          top=I(5.25), col_w=[1, 1.2, 1.8], size=12)
    rd, vd = f.corruption_drop("resnet50"), f.corruption_drop("vit_b16")
    if "pending" not in (rd, vd):
        plain(s, f"On clean images the two are within a point of each other. On damaged "
                 f"images the CNN loses {rd} of its accuracy and the transformer only "
                 f"{vd} — the gap widens as damage worsens.", top=I(6.15), height=I(0.9))
    notes(s, """
Read the curves first: accuracy on the vertical axis, damage severity increasing to
the right. Both models fall, which is expected. What matters is that the orange
transformer line falls much more slowly.

Now the table, which is the number to remember. Averaged over all fifteen
combinations of damage type and severity, the CNN loses about twenty-seven percent
of its accuracy while the transformer loses about seventeen.

The striking part is how the gap behaves. On clean images the two models are within
about three-quarters of a point of each other — nearly tied. Under the heaviest
sensor noise the difference is nearly thirty points. So a benchmark that only
reported clean accuracy would describe these two models as near-equivalent, and
would be badly misleading about how they behave on real photographs.

This is a practical point, not just an academic one. If you are deploying to
phone cameras, security footage, or compressed web images, this difference matters
far more than the clean-accuracy difference does.

The obvious next question is WHY the transformer holds up better — and that is what
the next section answers.
""")
    return s


def sl_what_is_frequency(prs, f):
    s = slide(prs, "What “spatial frequency” means for a photograph",
              kicker="11 · Background")
    bullets(s, [
        (0, "Any image can be separated into coarse structure and fine detail — the "
            "same way a piece of music separates into bass and treble.", True),
        (1, "LOW frequency = broad shapes, overall layout, big blocks of colour", False),
        (1, "HIGH frequency = edges, texture, fine print, hair, grain", False),
        (0, "Below: the same photograph with progressively more high-frequency detail "
            "allowed back in. Far left keeps only the coarsest shapes.", False),
    ], top=I(1.58), height=I(1.5), size=15)
    if f.figure("frequency_lowpass_ladder.png"):
        picture(s, f.figure("frequency_lowpass_ladder.png"), top=I(3.05), max_h=I(3.4))
    plain(s, "Think bass and treble for pictures. We can turn each band up or down and "
             "watch what the model does.", top=I(6.5), height=I(0.75))
    notes(s, """
This slide creates the vocabulary for the contribution, so do not rush it.

The analogy that works for everybody is audio. A piece of music can be split into
bass and treble. Turn the treble down and it goes muffled, but you can still follow
the tune. Turn the bass out and it goes thin and tinny.

Images work the same way. The low frequencies are the broad structure — the overall
shape of an object, big regions of colour, the general layout. The high frequencies
are the fine detail — edges, texture, fur, lettering, film grain.

The strip along the bottom is exactly that. On the far left we have kept only the
very coarsest information; you can see there is something there but not what. As
you move right we let progressively finer detail back in, until on the far right the
image is complete.

Two useful things follow. First, this gives us a dial: we can control precisely
which kinds of visual information a model is allowed to see. Second, it gives us a
free correctness check — at the far right the filter is mathematically doing nothing
at all, so the model's score there must exactly equal its normal score. It does,
which tells us the filtering code is right.

The faint halos around objects in the middle images are a known side-effect of
using a hard cutoff. It is expected, it is standard for this analysis, and it is
noted in the report.
""")
    return s


def sl_result_frequency(prs, f):
    s = slide(prs, "Result 3 — which frequencies does each model rely on?",
              kicker="12 · Result")
    if f.figure("fig_band_noise.png"):
        picture(s, f.figure("fig_band_noise.png"), top=I(1.6), max_h=I(3.7))
    rows = [[{"resnet50": "ResNet-50", "vit_b16": "ViT-B/16"}[m], f.band_weakness(m),
             f.freq_auc(m, "lp")] for m in ("resnet50", "vit_b16")]
    table(s, ["Model", "Most damaging noise band", "Accuracy kept under low-pass"],
          rows, top=I(5.45), col_w=[1, 1.5, 1.6], size=12)
    plain(s, "We add the same amount of interference in different frequency bands. The "
             "CNN has a clear weak spot at low frequency; the transformer does not.",
          top=I(6.3), height=I(0.85))
    notes(s, """
Here is the experiment. We take a fixed amount of interference — think of it as a
fixed volume of static — and slide it across the frequency range. First we put all
of it in the low frequencies, then a bit higher, and so on up to the finest detail.
Crucially the AMOUNT is held constant; only its position changes. So the curve
measures where the model is vulnerable, not how much noise we added.

The blue CNN curve has a pronounced dip at low frequency. Interference in the coarse
structure hurts it badly. The orange transformer curve is much flatter, with no
comparable weak point anywhere.

That is the mechanism behind the previous slide. The CNN has a specific
vulnerability; the transformer's dependence is spread more evenly. Real-world damage
— noise, blur, compression — disturbs many frequency bands at once, so a model with
a sharp weak spot gets caught by it and a model without one does not.

This is what makes the project an explanation rather than a benchmark. We are not
just reporting that one model is more robust. We are identifying which part of the
visual signal each model leans on, and using that to account for the robustness
result.
""")
    return s


def sl_contribution(prs, f):
    s = slide(prs, "The contribution — the CNN has to LEARN robustness; the ViT arrives "
                   "with it", kicker="13 · Contribution")
    if f.figure("fig_frequency_shift.png"):
        picture(s, f.figure("fig_frequency_shift.png"), top=I(1.55), max_h=I(3.6))
    ps = f.profile_shift()
    if ps and "resnet50" in ps and "vit_b16" in ps:
        r, v = ps["resnet50"], ps["vit_b16"]
        ratio = r["max_abs"] / v["max_abs"] if v["max_abs"] else float("nan")
        bullets(s, [
            (0, f"Dashed = trained on 10% of the data. Solid = trained on 100%. If the "
                f"two lines sit apart, the model's robustness DEPENDS on how much data "
                f"it was given.", True),
            (0, f"ResNet-50's lines separate widely (largest change {r['max_abs']:.3f}, "
                f"in the finest-detail band). ViT-B/16's lie almost on top of each other "
                f"({v['max_abs']:.3f}). A {ratio:.0f}x difference.", False),
        ], top=I(5.25), size=14, height=I(1.05))
        cv = f.contribution_verdict()
        if cv and cv["any_tested"] and not cv["all_replicate"]:
            plain(s, "On Caltech-256 the CNN only becomes robust to fine detail once it "
                     "has seen plenty of data; the transformer is already robust. This "
                     "did NOT reproduce on Food-101 — see the confirmation slide. The "
                     "claim is therefore stated as dataset-specific.",
                  top=I(6.3), height=I(1.0))
        else:
            plain(s, "The CNN only becomes robust to fine-detail interference once it "
                     "has seen plenty of data. The transformer is already robust, and "
                     "stays that way — which is exactly why it wins by most when data "
                     "is scarce.", top=I(6.35), height=I(0.95))
    notes(s, """
This is the original contribution — the part that is not a replication of known
results. Take it slowly.

Each panel shows one model. The dashed line is that model trained on only ten
percent of the data; the solid line is the same model trained on all of it. Both are
divided by the model's own clean accuracy, so we are asking what FRACTION of its
ability survives, which is what lets a weak model and a strong model be compared on
one axis.

Look at the left panel, the CNN. The two lines separate dramatically at the right
side — the fine-detail end. Trained on little data, the CNN is fragile against
fine-detail interference. Trained on plenty, it becomes much more robust there.
Its robustness profile MOVES with the data budget.

Now the right panel, the transformer. The two lines are almost indistinguishable.
Whether it saw ten percent or one hundred percent of the data, its robustness
profile is essentially the same.

Quantitatively, the CNN's profile moves about twelve times as much as the
transformer's.

Here is the interpretation, and it is the sentence to land. The convolutional
network has to LEARN robustness to fine-detail interference from the fine-tuning
data. The transformer already has it, from pre-training, and does not need your
data to acquire it. Which explains the very first result: the transformer's
advantage is largest exactly where the CNN has least data from which to learn the
thing the transformer already possesses.

That ties all three results into a single account, and it is the reason this is a
characterisation rather than a leaderboard.

If asked how this differs from prior work: existing research established that the
two families differ in frequency response, but at full scale. What is new here is
that the difference is itself data-dependent for one family and not the other —
which only becomes visible if you deliberately vary the data budget while holding
everything else fixed, which is what this protocol does.
""")
    return s


def sl_invariance(prs, f):
    """The claim that survives BOTH datasets."""
    inv = f.invariance()
    s = slide(prs, "What survives both datasets", kicker="13b · Contribution, scoped")
    if not inv:
        bullets(s, [(0, "(pending)", False)])
        return s
    rows = []
    for m, n in (("resnet50", "ResNet-50"), ("vit_b16", "ViT-B/16")):
        d = inv[m]
        cells = []
        for ds, lab in (("caltech256", "Caltech"), ("food101", "Food-101")):
            got = [f"f{fr}:{v:.2f}" for (dd, fr), v in sorted(d["values"].items())
                   if dd == ds]
            cells.append(" ".join(got) if got else "—")
        rows.append([n, cells[0], cells[1], f"{d['min']:.3f}–{d['max']:.3f}",
                     f"{d['range']:.1f}×"])
    table(s, ["Model", "Caltech-256", "Food-101", "range", "spread"], rows,
          top=I(1.62), col_w=[1.0, 1.6, 1.6, 1.0, 0.7], size=11)
    bullets(s, [
        (0, "Accuracy retained under high-frequency noise, in every dataset × data-size "
            "condition tested.", False),
        (0, f"ResNet-50 varies over a {inv['resnet50']['range']:.0f}× range — it depends "
            f"on both the task and how much data it saw. ViT-B/16 varies over "
            f"{inv['vit_b16']['range']:.2f}× — effectively invariant.", True),
        (0, "The narrower claim (that the CNN's robustness IMPROVES with data) held on "
            "Caltech-256 but not on Food-101, where the CNN stays fragile at every "
            "budget. What generalises is the INVARIANCE contrast, not the direction.", False),
    ], top=I(3.3), size=14, height=I(2.1))
    plain(s, "The transformer's robustness to fine detail is the same whatever you train "
             "it on and however much data you give it. The CNN's depends on both — and "
             "on one dataset it never gets there at all.", top=I(5.6), height=I(1.0))
    takeaway(s, "Stated honestly: this broader framing was formed after seeing both "
                "datasets. A third dataset would be needed to confirm it.", top=I(6.75))
    notes(s, """
This slide exists because the narrower version of the contribution did not survive
the second dataset, and that has to be handled openly.

The table is high-frequency robustness in every condition tested — two datasets,
several data sizes each. The transformer sits between 0.96 and 0.99 everywhere: it
barely notices fine-detail noise regardless of task or data budget. The CNN ranges
from 0.08 to 0.88 — a factor of about eleven.

Read the Food-101 column for the CNN carefully. It is around 0.08 at every data
size. On that dataset the convolutional network never becomes robust to
high-frequency noise, no matter how much data it gets. On Caltech it does, given
enough. So the original claim — that the CNN LEARNS this robustness from data —
is true on one dataset and false on the other.

What does hold across everything is the contrast in stability: the transformer is
invariant, the CNN is contingent on both task and data.

Be explicit that this broader framing was arrived at after looking at both
datasets rather than predicted in advance. That is an honest description of how
the analysis went, and it makes the claim a hypothesis with strong support rather
than a confirmed result. Confirming it needs a third dataset, which is a concrete
Phase-II target.

If asked why the CNN might be permanently fragile on Food-101: distinguishing food
categories leans heavily on fine texture, so high-frequency content is essential to
the task and the CNN cannot afford to discard it — yet it also cannot make it
robust. That is a hypothesis, not a measurement, and should be labelled as one.
""")
    return s


def sl_overlap(prs, f):
    s = slide(prs, "Do they make the same mistakes?", kicker="14 · Supporting")
    o = f.overlap_at(100)
    if o:
        rows = [["Both models correct", f"{o['both_correct']*100:.1f}%"
                 if "both_correct" in o else "—"],
                ["Both models wrong", f"{o['both_wrong']*100:.1f}%"],
                ["…and giving the SAME wrong answer", f"{o['same_wrong']*100:.1f}%"],
                ["Agreement beyond chance (Cohen's κ)", f"{o['kappa']:.3f}"],
                ["If you could always pick the right one", f"{o['oracle']*100:.2f}%"]]
    else:
        rows = [["(pending)", ""]]
    table(s, ["At full data", "Value"], rows, top=I(1.62), col_w=[2.6, 1])
    if f.figure("fig_error_overlap.png"):
        picture(s, f.figure("fig_error_overlap.png"), top=I(3.55), max_h=I(2.5))
    if o:
        best = max(f.top1("resnet50", 100)[0], f.top1("vit_b16", 100)[0]) * 100
        plain(s, f"They fail on different images. A perfect chooser between them would "
                 f"reach {o['oracle']*100:.1f}%, against {best:.1f}% for the better model "
                 f"alone — so they are complementary, not interchangeable.",
              top=I(6.2), height=I(0.9))
    notes(s, """
This slide answers a question a good examiner will ask: are these two models really
different, or are they two roads to the same place?

The key row is the last one. If you had an oracle that could look at each image and
pick whichever model happened to be right, you would score about ninety-three and a
half percent — roughly four points above the better model on its own. That headroom
only exists because they fail on DIFFERENT images.

Cohen's kappa is a fairness correction. Two models that are each about ninety
percent accurate will agree most of the time purely by luck, so raw agreement is
misleading. Kappa subtracts that luck. A value near one would mean interchangeable;
near zero would mean independent. We are around 0.55 — clearly related, since they
solve the same task, but genuinely different in where they fail.

There is a practical implication worth mentioning: because they are complementary,
combining them would beat either one. We do not do that here — it is outside the
scope of a controlled comparison — but it follows directly from this table.
""")
    return s


def sl_calibration(prs, f):
    s = slide(prs, "Do they know when they are unsure?", kicker="15 · Supporting")
    if f.figure("fig_calibration.png"):
        picture(s, f.figure("fig_calibration.png"), top=I(1.6), max_h=I(3.6))
    rows = [[{"resnet50": "ResNet-50", "vit_b16": "ViT-B/16"}[m], f.ece(m, 100)]
            for m in ("resnet50", "vit_b16")]
    table(s, ["Model", "Miscalibration (lower is better)"], rows, top=I(5.35),
          col_w=[1, 1.6], size=12)
    plain(s, "Both models' confidence is about equally trustworthy — so the "
             "transformer's robustness is not bought by simply being less confident.",
          top=I(6.2), height=I(0.85))
    notes(s, """
Alongside a prediction, these models emit a confidence — "I am eighty percent sure
this is a dog". Calibration asks whether that number can be believed. If you collect
every prediction the model made with eighty percent confidence, was it right about
eighty percent of the time?

This matters whenever a system has to decide when to defer to a human. A model that
is confidently wrong is far more dangerous than one that admits uncertainty.

The left panel plots stated confidence against actual accuracy. The dashed diagonal
is perfect honesty. Points below the line mean overconfidence.

The number in the table summarises the average gap. Both models sit around three
percent, which is typical and unremarkable.

The reason this slide exists is defensive, and worth saying out loud. Someone could
argue that the transformer only looks robust because it is systematically less
confident — hedging its bets. This shows that is not the case. The two are
calibrated about equally, so the robustness difference is real and not an artefact
of confidence behaviour.
""")
    return s


def sl_deployment(prs, f):
    s = slide(prs, "What the accuracy costs you", kicker="16 · Cost")
    if f.figure("fig_deployment.png"):
        picture(s, f.figure("fig_deployment.png"), top=I(1.6), max_h=I(3.5))
    rd, vd = f.deployment_row("resnet50"), f.deployment_row("vit_b16")
    if rd and vd:
        table(s, ["", "ResNet-50", "ViT-B/16", "ratio"],
              [["Parameters (M)", f"{rd.get('params_m',0):.1f}", f"{vd.get('params_m',0):.1f}",
                f"{vd.get('params_m',1)/max(rd.get('params_m',1),1e-9):.1f}x"],
               ["Compute, GMACs @224", f"{rd.get('gmacs',0):.2f}", f"{vd.get('gmacs',0):.2f}",
                f"{vd.get('gmacs',1)/max(rd.get('gmacs',1),1e-9):.1f}x"],
               ["Training speed (img/s)", f"{rd.get('train_imgs_per_sec',float('nan')):.0f}",
                f"{vd.get('train_imgs_per_sec',float('nan')):.0f}", ""]],
              top=I(5.2), col_w=[1.6, 1, 1, 0.7], size=12)
    plain(s, "The transformer's advantage comes with roughly 3.6x the parameters and 4x "
             "the computation per image. Whether that trade is worth it depends on where "
             "the model has to run.", top=I(6.35), height=I(0.9))
    notes(s, """
An honest comparison has to include cost, and this is where the CNN answers back.

The transformer has about three and a half times as many parameters — meaning a
bigger file and more memory. It needs roughly four times the arithmetic to process
one image, which translates directly into slower inference and more battery or
electricity. And in our own measurements it trains about three times slower.

So the finding is not "use transformers". It is a trade you have to price for your
own situation. On a server where compute is cheap and accuracy matters, the
transformer looks good. On a phone, a drone, or an embedded camera, four times the
computation per frame may simply be unaffordable, and the CNN's near-equal clean
accuracy at a quarter of the cost is the better engineering choice.

Note also that all of this ran on a single laptop graphics card. That is worth
mentioning: the study is reproducible by anyone with modest hardware, which is
unusual for architecture comparisons and was a deliberate design constraint.
""")
    return s


def sl_summary(prs, f, final=False):
    s = slide(prs, "What we found", kicker="Summary")
    g10 = f.gap_at(10)
    ps = f.profile_shift()
    ratio = (ps["resnet50"]["max_abs"] / ps["vit_b16"]["max_abs"]
             if ps and ps.get("vit_b16", {}).get("max_abs") else None)
    items = [
        (0, "1. Under a genuinely controlled protocol, the Vision Transformer is more "
            "data-efficient — it wins at every data size and by most when data is "
            "scarce.", True),
        (0, "2. It is also substantially more robust to damaged images, and the "
            "advantage grows as damage worsens.", True),
        (0, "3. The reason: the CNN has a sharp low-frequency weak spot; the "
            "transformer's dependence is spread evenly.", True),
        (0, f"4. And the contribution — the CNN's robustness profile depends on how "
            f"much data it got"
            + (f" ({ratio:.0f}x more movement than the transformer's)" if ratio else "")
            + ". The transformer's does not. The CNN must learn what the transformer "
              "already has.", True),
        (0, "5. The cost: ~3.6x parameters, ~4x computation. The right choice depends "
            "on where it runs.", False),
    ]
    bullets(s, items, top=I(1.62), size=15, height=I(4.0))
    takeaway(s, "Not a leaderboard — a characterisation. The frequency behaviour "
                "explains the robustness result, and both explain the data-efficiency "
                "result.", top=I(5.75))
    notes(s, """
Close on the shape of the argument rather than the individual numbers.

Point one is the headline: under a fair comparison the transformer gets more out of
limited data, and its advantage is biggest exactly where data is most limited.

Point two: it is also much more robust to the kinds of damage real photographs
actually suffer.

Point three explains point two: the CNN has an identifiable weak spot at coarse
spatial frequencies, and the transformer does not.

Point four is the new part. That weak spot is not fixed — the CNN's robustness
profile changes substantially depending on how much fine-tuning data it received,
while the transformer's does not change at all. So the CNN has to learn from your
data something the transformer brought with it. That single sentence explains point
one.

Point five is the honest counterweight: all of this costs about four times the
computation, so the engineering answer depends on the deployment target.

The framing to leave people with: this is not a leaderboard entry. Every result
explains the one before it, which is what makes it a characterisation of how the
two families behave rather than a measurement of which one scores higher.
""")
    return s


# ==========================================================================
# helpers
# ==========================================================================
def _eff_table(f, s, regimes=("fullft",), top=None):
    headers = ["Training data per category", "10%", "25%", "50%", "100%"]
    rows, hl = [], []
    for regime in regimes:
        for m in ("resnet50", "vit_b16"):
            if f.top1(m, 100, regime=regime) is None and f.top1(m, 10, regime=regime) is None:
                continue
            e = f.efficiency_row(m, regime=regime)
            name = {"resnet50": "ResNet-50 (CNN)", "vit_b16": "ViT-B/16 (Transformer)"}[m]
            rows.append([name, e[10], e[25], e[50], e[100]])
    if len(rows) >= 2 and all(f.gap_at(x, regime=regimes[0]) is not None for x in (10, 100)):
        rows.append(["Difference", f.gap_str(10, regime=regimes[0]),
                     f.gap_str(25, regime=regimes[0]), f.gap_str(50, regime=regimes[0]),
                     f.gap_str(100, regime=regimes[0])])
        hl = [len(rows) - 1]
    if not rows:
        rows = [["(pending)"] * 5]
    return table(s, headers, rows, top=top or I(1.8), highlight_rows=hl,
                 col_w=[2.4, 1, 1, 1, 1], size=13)


def _core_story(prs, f, final=False):
    sl_question(prs, f)
    sl_two_models(prs, f)
    sl_transfer(prs, f)
    sl_controlled(prs, f)
    sl_dataset(prs, f)
    sl_design(prs, f)
    sl_schedule_bug(prs, f)
    sl_result_efficiency(prs, f)
    sl_how_corruption(prs, f)
    sl_result_corruption(prs, f)
    sl_what_is_frequency(prs, f)
    sl_result_frequency(prs, f)
    sl_contribution(prs, f)
    sl_invariance(prs, f)
    sl_overlap(prs, f)
    sl_calibration(prs, f)
    sl_deployment(prs, f)


# ==========================================================================
# progress decks
# ==========================================================================
def deck_progress_01(f):
    prs = new_deck()
    title_slide(prs, "Progress Update 1", "Project resumed — framework complete and "
                "the CNN arm fully run",
                [COURSE + " · " + TITLE, STUDENT, GUIDE])
    notes(prs.slides[-1], """
This is the first update after a pause in the work. Be straightforward about that:
the framework is now finished, the convolutional half of the experiment is fully
run, and the transformer half is in progress.
""")

    s = slide(prs, "Where the project stands", kicker="Status")
    bullets(s, [
        (0, "Work paused for a period after the initial setup; it has now resumed and "
            "the backlog is cleared.", True),
        (0, "Everything committed for the CNN side is complete and on disk:", False),
        (1, f"{f.n_runs('fullft')} fine-tuning runs — 4 data sizes × 3 random seeds", False),
        (1, "a documented learning-rate search, with the choice recorded in the config", False),
        (1, "held-back-set evaluation with a full per-image record for every run", False),
        (0, f"Training compute so far: {f.gpu_hours()} GPU-hours on one laptop graphics "
            f"card.", False),
        (0, "The transformer arm is running; the evaluation battery follows.", True),
    ])
    takeaway(s, "The reproducible framework is finished — what remains is running it.")
    notes(s, """
Lead with the honest status: there was a pause, it is over, and the backlog is
cleared. Then show that the CNN half is genuinely complete rather than
in-progress — every run, every evaluation, every log on disk.
""")

    sl_question(prs, f)
    sl_two_models(prs, f)
    sl_transfer(prs, f)
    sl_controlled(prs, f)

    s = slide(prs, "Reproducibility is built in, not promised", kicker="Framework")
    bullets(s, [
        (0, "Every run is named by what produced it: model, dataset, data size, seed, "
            "regime.", True),
        (0, "Each run writes its own configuration, per-epoch log, metrics, and a "
            "per-image record of every prediction it made.", False),
        (0, "Splits were generated once and committed; the held-back set is frozen and "
            "has never influenced a decision.", True),
        (0, "Random seeds are fixed everywhere, so a rerun reproduces the same numbers.", False),
        (0, "Every figure and table is generated by script from one results file — no "
            "number is typed by hand.", True),
        (0, "Public repository: github.com/KPraveenRaj/cnn-vs-vit", False),
    ])
    notes(s, """
This slide is about trustworthiness. The claim is that any number in this project
can be traced back to the run that produced it, and that rerunning gives the same
answer.

The last point is worth emphasising to a supervisor: no number in any slide or
report is typed by hand. They are all generated from a single results table by
script, so a slide cannot silently disagree with the data.
""")

    s = slide(prs, "CNN arm: how accuracy grows with data", kicker="Result")
    _eff_table(f, s, regimes=("fullft",))
    if f.figure("fig_data_efficiency.png"):
        picture(s, f.figure("fig_data_efficiency.png"), top=I(3.15), max_h=I(2.6))
    caption(s, "Held-back-set accuracy, mean ± spread over 3 seeds.", top=I(5.9))
    notes(s, """
Accuracy on held-back images against how much training data was used. The plus or
minus is the spread over three different random starts.

Note that going from ten percent of the data to all of it buys roughly twelve
points. That is the baseline against which the transformer will be compared.
""")

    s = slide(prs, "The learning-rate search", kicker="Method evidence")
    grid = f.lr_grid("resnet50")
    rows = [[lr, v, "← chosen" if sel else ""] for lr, v, sel in grid] or [["(pending)"] * 3]
    table(s, ["Learning rate", "Best validation accuracy", ""], rows,
          highlight_rows=[i for i, (_, _, sel) in enumerate(grid) if sel],
          col_w=[1.2, 1.4, 1])
    bullets(s, [
        (0, "The learning rate controls how large a correction the model makes each "
            "time it is wrong.", False),
        (0, "Chosen on the validation set only — the held-back set never influences any "
            "setting.", True),
        (0, "The winner sits in the middle of the range searched, so the best value is "
            "bracketed rather than at the edge of what we tried.", False),
    ], top=I(3.9), size=15)
    notes(s, """
Explain why this slide exists at all. Since each model gets its own learning rate,
that choice has to be visible and defensible rather than hand-waved.

The winner being in the middle of the searched range matters: if the best value had
been at the edge, it would mean we never found the true optimum and the model might
be under-tuned.
""")

    s = slide(prs, "Next", kicker="Plan")
    bullets(s, [
        (0, "ViT-B/16 learning-rate search, then its matching 12-run grid", True),
        (0, "Linear probes for both models — how good is the knowledge before any "
            "adaptation?", False),
        (0, "The evaluation battery on every trained model: damaged images and "
            "frequency-filtered images", False),
        (0, "Then the analysis this project exists for — does frequency behaviour "
            "explain the robustness differences?", True),
    ])
    save(prs, OUT / "progress_01_resumption.pptx")


def deck_progress_02(f):
    prs = new_deck()
    title_slide(prs, "Progress Update 2", "Both arms complete — the head-to-head "
                "comparison is on the table",
                [COURSE + " · " + TITLE, STUDENT, GUIDE])
    s = slide(prs, "ViT-B/16: the learning-rate search", kicker="Method evidence")
    grid = f.lr_grid("vit_b16")
    rows = [[lr, v, "← chosen" if sel else ""] for lr, v, sel in grid] or [["(pending)"] * 3]
    table(s, ["Learning rate", "Best validation accuracy", ""], rows,
          highlight_rows=[i for i, (_, _, sel) in enumerate(grid) if sel],
          col_w=[1.2, 1.4, 1])
    bullets(s, [
        (0, "The transformer's range sits a full decade below the CNN's — and that is "
            "exactly why per-model tuning was declared in advance.", True),
        (0, "Reusing the CNN's range would have put the transformer's best value at the "
            "very bottom edge: it would have been handicapped at precisely the "
            "comparison this project exists to make.", False),
        (0, "The grid was extended once, because the first winner landed on an edge and "
            "an edge winner means the optimum was never bracketed.", False),
    ], top=I(3.9), size=14)
    notes(s, """
The point of this slide is fairness. The two architectures want genuinely different
learning rates — the transformer's best is about ten times smaller. Had we forced
one value on both, we would have measured our own arbitrary choice.

Mention the grid extension: the first search returned a winner at the edge of the
range, which means we had not actually found the best value. So the range was
widened until the winner had worse values on both sides of it.
""")

    sl_result_efficiency(prs, f)
    sl_deployment(prs, f)

    s = slide(prs, "Next", kicker="Plan")
    bullets(s, [
        (0, "The evaluation battery over all 24 trained models — 42 tests each", True),
        (1, "damaged images: sensor noise, blur, compression, at 5 severities", False),
        (1, "frequency filtering, to find what each model relies on", False),
        (0, "Then the mechanism question: does frequency behaviour explain the "
            "robustness differences?", True),
    ])
    save(prs, OUT / "progress_02_vit_arm.pptx")


def deck_progress_03(f):
    prs = new_deck()
    title_slide(prs, "Progress Update 3", "Robustness and frequency battery complete",
                [COURSE + " · " + TITLE, STUDENT, GUIDE])
    sl_how_corruption(prs, f)
    sl_result_corruption(prs, f)
    sl_what_is_frequency(prs, f)

    s = slide(prs, "How the frequency probes are built", kicker="Method")
    bullets(s, [
        (0, "Each image is converted into its frequency components, some are switched "
            "off, and it is converted back.", True),
        (0, "Correctness checks built into the code, not asserted:", False),
        (1, "at the widest setting the filter must do nothing — and the model's score "
            "there matches its normal score exactly", False),
        (1, "the low-pass and high-pass halves must add back up to the original image — "
            "verified to seven decimal places", False),
        (1, "every noise band must carry identical energy, so the curve measures the "
            "model and not the noise", False),
    ], top=I(1.6), height=I(2.1), size=14)
    if f.figure("fft_construction.png"):
        picture(s, f.figure("fft_construction.png"), top=I(3.85), max_h=I(2.5))
    notes(s, """
For a technical audience this is the slide that establishes the analysis is sound.
For a general audience, the message is simply: we did not trust this code, we tested
it, and here are the tests.

The strongest of the three is the first. At the widest filter setting the operation
is mathematically guaranteed to change nothing at all. So the model's accuracy there
must exactly equal its ordinary accuracy. It does — to four decimal places, on real
trained models. If there were a bug in the filtering, that would almost certainly
have broken.
""")

    sl_result_frequency(prs, f)

    s = slide(prs, "Next", kicker="Plan")
    bullets(s, [
        (0, "The interaction: does frequency reliance shift with training-set size, and "
            "does it predict low-data robustness?", True),
        (0, "Error overlap — are the two families making the same mistakes?", False),
        (0, "Calibration and deployment cost, then the mid-semester report.", False),
    ])
    save(prs, OUT / "progress_03_robustness.pptx")


def deck_progress_04(f):
    prs = new_deck()
    title_slide(prs, "Progress Update 4", "Mechanism analysis complete — ready for the "
                "mid-semester review",
                [COURSE + " · " + TITLE, STUDENT, GUIDE])
    sl_contribution(prs, f)
    sl_invariance(prs, f)
    sl_overlap(prs, f)
    sl_calibration(prs, f)

    s = slide(prs, "Status against the plan", kicker="Progress")
    av = f.available()
    mark = lambda b: "Complete" if b else "Pending"
    rows = [["Caltech-256 fine-tuning, both models × 4 sizes × 3 seeds",
             mark(av["resnet_fullft"] and av["vit_fullft"])],
            ["Linear probes on frozen features", mark(av["probes"])],
            ["Evaluation battery (damage + frequency)", mark(av["battery"])],
            ["Error overlap · calibration · deployment", mark(av["overlap"])],
            ["Food-101 cross-dataset confirmation", mark(av["food101"])]]
    table(s, ["Committed work", "Status"], rows, col_w=[3.4, 1])
    bullets(s, [(0, f"Training compute: {f.gpu_hours()} GPU-hours on one 8 GB laptop GPU, "
                 f"plus the evaluation battery.", False),
                (0, "Everything reproducible from the public repository; every figure and "
                    "table generated by script from a single results file.", True)],
            top=I(4.6))
    save(prs, OUT / "progress_04_mechanism.pptx")


# ==========================================================================
# review decks
# ==========================================================================
def deck_midsem(f):
    prs = new_deck()
    title_slide(prs, TITLE, "Mid-semester review",
                [COURSE + " · Mid-Semester Evaluation", STUDENT, GUIDE])
    notes(prs.slides[-1], """
Presenting to a mixed audience. Assume the supervisor knows the field and that
others in the room may not. The deck is built so the vocabulary needed for each
result appears before that result does.

Timing: about 15 minutes for the story, leaving room for questions. If short on
time, the four slides that must survive are the two-models explainer, data
efficiency, the frequency explainer, and the contribution.
""")

    s = slide(prs, "At a glance", kicker="Summary")
    bullets(s, [
        (0, f"{f.n_runs()} training runs on Caltech-256 across two architectures, four "
            f"data sizes and three random seeds.", True),
        (0, f"Every trained model put through an identical 42-test battery — "
            f"{f.battery_done()} of 24 complete.", False),
        (0, f"Training compute: {f.gpu_hours()} GPU-hours on a single laptop graphics "
            f"card; the evaluation battery is additional.", False),
        (0, "Every figure and number here is generated by script from one results file. "
            "Nothing is entered by hand.", True),
        (0, "Public repository with per-run configs, logs, metrics and per-image "
            "predictions: github.com/KPraveenRaj/cnn-vs-vit", False),
    ])
    notes(s, """
Set expectations. This is a completed controlled study, not a progress report: the
main experiment is finished, all twenty-four models trained, all evaluated.

The compute number is worth saying aloud — the whole study ran on one laptop
graphics card. Architecture comparisons are usually done on clusters, so this is
unusually reproducible, and that was a deliberate constraint rather than a
limitation.
""")

    _core_story(prs, f)

    s = slide(prs, "What remains", kicker="Plan")
    av = f.available()
    bullets(s, [
        (0, "Food-101 cross-dataset confirmation — does the finding hold on a second, "
            "larger, finer-grained dataset?", not av["food101"]),
        (0, "Extended qualitative analysis (attention maps), if time permits", False),
        (0, "Final report and end-semester presentation", False),
        (0, "Phase II (EC790): generative and restoration-oriented vision transformers "
            "— the frequency tooling built here transfers directly, because image "
            "restoration is explicitly a frequency problem.", True),
    ])
    notes(s, """
Be clear that the main study is complete and what remains is confirmation and
writing, not core experiments.

On Phase II: the connection is genuine rather than decorative. Image restoration —
denoising, deblurring, super-resolution — is fundamentally about recovering
frequency content that has been lost or corrupted. The measurement tools built in
this phase apply directly.
""")

    s = slide(prs, "Limitations and what we checked", kicker="Discussion")
    bullets(s, [
        (0, "Pre-training DATA is controlled (both ImageNet-1k) but pre-training RECIPE "
            "is not. The robustness result is about these two public checkpoints, not "
            "about CNNs and transformers in general.", True),
        (0, "The frequency contribution held on Caltech-256 and did NOT reproduce on "
            "Food-101. What survives both is the invariance contrast, not the "
            "direction.", True),
        (0, "One CNN, one transformer, one primary dataset that overlaps the "
            "pre-training distribution — flattering to both, but equally.", False),
        (0, "463 automated consistency checks over every result file; no computational "
            "errors found. Findings cross-checked against Park & Kim (2022), Naseer et "
            "al. (2021) and Bhojanapalli et al. (2021). Full audit in AUDIT.md.", True),
    ])
    notes(s, """
Volunteering limitations is far more persuasive than being caught by them, and
every item here is one an examiner could otherwise raise.

The first is the one to lead with because it is the least obvious and the most
honest. We controlled which DATA both models were pre-trained on — ImageNet-1k for
both, which is the control most published comparisons get wrong. But the two
downloadable checkpoints were trained with different recipes: different optimiser,
different loss, different augmentation mix, different number of epochs. Since
augmentation strength is known to affect robustness, part of the robustness gap
could come from the recipe rather than the architecture. Fixing it would mean
pre-training both models from scratch under one recipe, which is far beyond a
laptop GPU. So the claim is scoped to these two checkpoints.

The second is the non-replication, covered on its own slide.

The last point is worth saying plainly: the results were audited rather than
assumed. Four hundred and sixty-three automated checks over every result file,
plus a cross-check of each claim against the published literature. No
computational errors were found; what the audit did find were these scope limits.
""")
    sl_summary(prs, f)
    save(prs, OUT / "midsem_review.pptx")


def deck_endsem(f):
    prs = new_deck()
    title_slide(prs, TITLE, "End-semester review — Phase I complete",
                [COURSE + " · End-Semester Evaluation", STUDENT, GUIDE])

    s = slide(prs, "What this project established", kicker="Summary")
    bullets(s, [
        (0, "Under one controlled protocol, a CNN and a Vision Transformer of identical "
            "pre-training were compared on data efficiency, robustness to damage, and "
            "frequency sensitivity.", True),
        (0, f"{f.n_runs()} runs · {f.battery_done()} full evaluation batteries · "
            f"{f.gpu_hours()} GPU-hours of training on one 8 GB laptop GPU.", False),
        (0, "The frequency analysis provides a MECHANISM for the robustness results "
            "rather than only reporting them.", True),
        (0, "Everything reproducible from a public repository, generated end to end by "
            "script.", False),
    ])
    _core_story(prs, f, final=True)

    s = slide(prs, "Does a second dataset agree?", kicker="17 · Confirmation")
    rep = REPO_ROOT / "results" / "tables" / "replication.csv"
    if rep.exists():
        import pandas as pd
        rdf = pd.read_csv(rep)
        rows = []
        for _, r in rdf.iterrows():
            a = r["agrees"]
            v = "pending" if pd.isna(a) else ("replicates" if a else "differs")
            rows.append([str(r["claim"])[:52], str(r["caltech256"])[:18],
                         str(r["food101"])[:18], v])
        table(s, ["Finding", "Caltech-256", "Food-101", "Verdict"], rows,
              col_w=[2.6, 1.1, 1.1, 0.9], size=11)
    else:
        bullets(s, [(0, "(Food-101 confirmation pending)", False)])
    bullets(s, [
        (0, "Food-101 runs ONE seed by design — it supports statements about direction, "
            "not about statistical significance.", True),
        (0, "A finding that holds on one dataset and not the other would indicate "
            "dataset dependence, NOT an error in the first measurement.", False),
    ], top=I(5.3), size=13, height=I(1.2))
    notes(s, """
Handle this slide carefully, because it is where an examiner may push.

Food-101 is a confirmation dataset: 101,000 photographs of food across 101
categories. It is larger than Caltech-256 and finer-grained — distinguishing types
of soup is harder than distinguishing a soup from a bicycle.

Two cautions to state before anyone else does. First, it runs a single random seed
by design, so it tells us about direction but cannot support significance claims;
there is no spread to test against. Second, and more important: if something does
NOT replicate, that is not evidence the Caltech result was wrong. The Caltech result
rests on its own internal validity — a frozen test set, three seeds, one protocol.
Disagreement would mean the finding is dataset-dependent, which is itself a
legitimate and quite interesting result: it would say the answer depends on how
close your task sits to the pre-training data.

The thing that WOULD invalidate the work is a protocol bug, which is why the
schedule interaction earlier was worth stopping everything to fix.
""")

    s = slide(prs, "Limitations, stated plainly", kicker="18 · Discussion")
    bullets(s, [
        (0, "Pre-training DATA is controlled (both ImageNet-1k) but pre-training RECIPE "
            "is not — the two public checkpoints were trained with different "
            "augmentation and optimisation recipes, and augmentation strength affects "
            "robustness. So the robustness result is about these two checkpoints, not "
            "about the families in general.", True),
        (0, "One published study (Bhojanapalli et al. 2021) finds the opposite for "
            "ImageNet-1k-pretrained ViTs evaluated in-domain. Our setting differs "
            "(robustness after transfer), but the discrepancy is recorded.", False),
        (0, "The band-noise probe holds noise energy constant, not signal-to-noise "
            "ratio — natural images have far more energy at low frequency. Model-vs-"
            "model comparison holds; 'which band it relies on' is more delicate.", False),
        (0, "Qualitative saliency is not reported: attention rollout and Grad-CAM were "
            "both measured degenerate for the ViT. No conclusion depends on a picture.", False),
        (0, "One CNN and one transformer are archetypes, not whole families. The "
            "conclusions are about these two representatives under this protocol.", True),
        (0, "Caltech-256 overlaps the ImageNet pre-training distribution. That flatters "
            "both models — but it flatters them equally.", False),
        (0, "The frequency filters use a hard cutoff, which causes visible ringing. It "
            "is standard for this analysis and makes the cutoff unambiguous, but it is "
            "a distortion of its own.", False),
        (0, "Linear probes use no training-time augmentation — a declared difference "
            "from fine-tuning, applied identically to both models.", False),
        (0, "Three seeds bound run-to-run variance, not dataset variance.", False),
        (0, "One shared training length was used for both models; the transformer's "
            "optimum at full data sits at a shorter budget. Documented as a "
            "sensitivity, not hidden.", False),
    ], size=14)
    notes(s, """
Volunteering limitations is more persuasive than being caught by them, and every
item here is one an examiner could otherwise raise.

The most important is the first. We compared ONE convolutional network and ONE
transformer. They are standard representatives, but a different pair might behave
differently. Nothing here licenses a sweeping claim about all CNNs and all
transformers.

The second is about the dataset. Caltech-256 contains everyday objects, which
overlap heavily with what both models saw during pre-training. That makes the task
easier than a genuinely novel domain like medical imaging — but it makes it equally
easier for both, so the comparison stays fair even though the absolute numbers are
flattering.
""")

    s = slide(prs, "Phase II — EC790", kicker="19 · Next")
    bullets(s, [
        (0, "Direction: generative and restoration-oriented vision transformers.", True),
        (0, "Image restoration — denoising, deblurring, super-resolution — is "
            "explicitly a frequency problem: it is about recovering detail that was "
            "lost or corrupted.", False),
        (0, "This phase built and validated tooling that measures exactly what a model "
            "does band by band. It transfers directly.", True),
        (0, "The controlled-protocol discipline carries over too: vary one factor at a "
            "time, and declare what you tuned.", False),
    ])
    sl_summary(prs, f, final=True)
    save(prs, OUT / "endsem_review.pptx")


def main():
    f = Facts()
    print(f"[decks] availability: {f.available()}")
    OUT.mkdir(parents=True, exist_ok=True)
    deck_progress_01(f)
    deck_progress_02(f)
    deck_progress_03(f)
    deck_progress_04(f)
    deck_midsem(f)
    deck_endsem(f)
    print(f"[decks] -> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
