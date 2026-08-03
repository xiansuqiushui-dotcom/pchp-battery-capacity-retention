# V383 engineering-metadata claim verification

## Verification rule

Engineering characteristics in Supplementary Tables S1--S2 were transcribed from the frozen processed metadata used by the scoring workflows and checked against the cited dataset records. Missing or non-reconstructable fields are reported as NR; no chemistry, temperature, rate, or state-of-charge value was inferred from model behavior.

## Source routing

| Claim group | Evidence route | Manuscript treatment |
|---|---|---|
| Development-domain counts and protocol metadata | Frozen BatteryLife processed tables and archived processing contract | Cited to the BatteryLife article and processed snapshot. |
| Oxford LFP | Frozen external table plus Wheeler et al. dataset record | Chemistry, form factor, nominal capacity, temperature, and rate/SOC fields reported only where available. |
| Imperial/Kirkaldy NMC | Frozen external table plus Kirkaldy et al. dataset record | Mixed duty-cycle and temperature ranges retained as ranges. |
| Luh NMC--SiO$_x$ | Frozen external table plus Luh et al. dataset record | Zero-rate entries explicitly interpreted as non-cycling/calendar conditions. |
| ILCC varying usage | Frozen external table plus Li et al. dataset record | Protocol diversity and pouch-cell capacity retained without generalizing beyond the evaluated roster. |
| Stanford calendar aging | Frozen external table plus Lam et al. dataset record | Nominal-capacity range marked as derived from the frozen ingestion contract rather than a single manufacturer specification. |
| Multistage 50E | Frozen external table plus Stroebl et al. dataset record | Partial-SOC and multistage conditions reported as evaluated metadata. |

## Wording audit

- “Temperature” is qualified as measured early-charge temperature or programmed ambient temperature according to source availability.
- “Nominal capacity” is not presented as measured capacity.
- Rate and SOC ranges describe protocols entering the scored records, not every experiment in each source archive.
- NR is used instead of guessing.
