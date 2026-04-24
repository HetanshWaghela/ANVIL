# Synthetic Pressure Equipment Standard (SPES-1)

*A fictional standard mirroring the ASME BPVC Section VIII Div. 1 structure for use in anvil.*

---

## Part A — General Requirements

### A-23 Maximum Allowable Stress Values

(a) The maximum allowable stress values for materials used in the construction of pressure vessels shall be taken from Part M of this Standard. Specifically, allowable stress values for carbon and low-alloy steels shall be obtained from Table M-1 (see Part M).

(b) Allowable stress values are tabulated at discrete design temperatures. When a design temperature falls between two tabulated values, linear interpolation shall be used, in accordance with A-23(c).

(c) No extrapolation beyond the tabulated temperature range is permitted. If the design temperature exceeds the maximum tabulated temperature for a material, that material shall not be used.

### A-25 Corrosion

(a) A corrosion allowance (CA) shall be added to the minimum required thickness calculated by A-27 or A-32.

(b) The corrosion allowance shall be specified in the design datasheet and shall not be taken as less than zero.

(c) When back-calculating the Maximum Allowable Working Pressure (MAWP), the effective corroded thickness equals the nominal thickness minus the corrosion allowance.

### A-27 Thickness of Shells Under Internal Pressure

(a) The minimum required thickness of shells subjected to internal pressure shall not be less than that computed by the following formulas, per A-27(c) and A-27(d).

(b) The allowable stress S shall be obtained per A-23, and the joint efficiency E shall be obtained from Table B-12 (see B-12).

(c) Cylindrical Shells.

**A-27(c)(1) — Inside Radius Formula:**

```
t = (P × R) / (S × E − 0.6 × P)
```

where:

- t = minimum required thickness, mm
- P = design pressure, MPa
- R = inside radius of the shell course, mm
- S = maximum allowable stress at design temperature, MPa (see Table M-1)
- E = joint efficiency (see Table B-12)

This formula applies when: t ≤ R/2 AND P ≤ 0.385 × S × E. When these conditions are not satisfied, the rules of A-27(c)(3) shall be applied (not covered here).

**A-27(c)(2) — Outside Radius Formula:**

```
t = (P × Ro) / (S × E + 0.4 × P)
```

where Ro is the outside radius of the shell course. Applicable when P ≤ 0.385 × S × E.

(d) Spherical Shells.

**A-27(d) — Spherical Shell Formula:**

```
t = (P × R) / (2 × S × E − 0.2 × P)
```

Applicable when t ≤ 0.356 × R AND P ≤ 0.665 × S × E.

### A-32 Formed Heads

Formed heads (ellipsoidal, torispherical, hemispherical, conical) are covered by this paragraph. For a 2:1 ellipsoidal head:

```
t = (P × D) / (2 × S × E − 0.2 × P)
```

where D is the inside diameter. The joint efficiency E shall be taken from Table B-12 and the allowable stress S from Table M-1 per A-23.

---

## Part B — Welding Requirements

### B-12 Joint Efficiency

(a) The joint efficiency E used in the thickness formulas of A-27 and A-32 shall be taken from Table B-12 as a function of the joint type (Types 1 through 6) and the extent of radiographic examination (Full RT, Spot RT, or None).

(b) Table B-12 — Joint Efficiency Values:

| Joint Type | Description | Full RT | Spot RT | No RT |
|---|---|---|---|---|
| 1 | Double-welded butt, full penetration | 1.00 | 0.85 | 0.70 |
| 2 | Single-welded butt with backing strip | 0.90 | 0.80 | 0.65 |
| 3 | Single-welded butt without backing | 0.80 | 0.70 | 0.60 |
| 4 | Double full-fillet lap | 0.75 | 0.65 | 0.55 |
| 5 | Single full-fillet lap with plug welds | 0.70 | 0.60 | 0.50 |
| 6 | Single full-fillet lap without plug welds | 0.65 | 0.55 | 0.45 |

(c) The radiography level selected shall be consistent with the service conditions and shall be documented.

---

## Part M — Material Properties

### M-1 Table M-1 — Allowable Stress Values for Carbon and Low-Alloy Steels

The following table gives maximum allowable stress values S (MPa) for selected synthetic material specifications at tabulated design temperatures. See A-23 for use. For materials not listed, consult Part M supplementary tables.

| Spec | Grade | Product Form | 40°C | 100°C | 150°C | 200°C | 250°C | 300°C | 350°C | 400°C | 450°C | 500°C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SM-516 | Gr 70 | Plate | 138 | 138 | 138 | 134 | 127 | 121 | 114 | 103 | 73 | 49 |
| SM-515 | Gr 70 | Plate | 138 | 138 | 138 | 132 | 124 | 118 | 110 | 98 | 69 | 46 |
| SM-106 | Gr B | Pipe | 118 | 118 | 118 | 115 | 108 | 102 | 96 | 86 | 61 | 41 |
| SM-105 | — | Forging | 138 | 138 | 138 | 133 | 125 | 119 | 112 | 100 | 71 | 47 |
| SM-240 | Type 304 | Plate | 137 | 127 | 119 | 113 | 108 | 103 | 100 | 97 | 95 | 93 |
| SM-240 | Type 316 | Plate | 137 | 137 | 130 | 123 | 117 | 112 | 108 | 105 | 103 | 101 |

### M-23 Material Restrictions

(a) SM-516 Gr 70 shall not be used at design temperatures exceeding 500°C.

(b) SM-106 Gr B shall not be used at design temperatures exceeding 500°C and is not permitted for cyclic service above 400°C without additional analysis.

---

## Appendix — Worked Example (Informative)

For a cylindrical shell with P = 1.5 MPa, inside diameter 1800 mm, design temperature 350°C, material SM-516 Gr 70, Type 1 joint with full RT, and 3.0 mm corrosion allowance, the minimum required thickness per A-27(c)(1) is computed using S = 114 MPa (Table M-1) and E = 1.00 (Table B-12). After adding corrosion allowance and rounding to the next standard plate, the design thickness is selected.
