You are a research and Python data-analysis agent.

Research question:
In South Korea, how have workplace-aspiration concentration and alcohol
consumption changed over time, and is there descriptive evidence that
they move together?

Treat this as an exploratory association study, not a causal test.
Do not assume that South Korea is a monoculture or that concentrated
aspirations necessarily cause alcohol consumption.

PRIMARY DELIVERABLE
Create a reproducible Python script that loads verified data, computes
career-aspiration concentration, and plots both series over time.
Also provide tidy CSVs, source metadata, and a short limitations report.

1. DATA FEASIBILITY FIRST

Find actual downloadable datasets and inspect their metadata before
choosing a study period. Do not invent table IDs, API endpoints,
historical observations, survey frequencies, or category mappings.

Career-data starting points:
- KOSIS official portal: https://kosis.kr/eng/
- Statistics Korea Social Survey.
- Youth Statistics publications and their underlying source tables.
- Korea Employment Information Service Youth Panel as an alternative.

Search in English and Korean, including:
"선호하는 직장", "청년 선호 직장", "사회조사", "청소년 통계".

Alcohol-data starting points:
- OECD alcohol-consumption indicator:
  https://www.oecd.org/en/data/indicators/alcohol-consumption.html
- WHO/World Bank indicator SH.ALC.PCAP.LI:
  https://databank.worldbank.org/metadataglossary/health-nutrition-and-population-statistics/series/SH.ALC.PCAP.LI
- KDCA/KNHANES, if compatible age-specific drinking rates can be found.

For every candidate source, verify:
- Observation years versus publication years.
- Population and age range.
- Single-choice versus multiple-response questions.
- Exact categories and question wording.
- Survey weights and sample sizes where available.
- Missing-value and nonresponse treatment.
- Definition changes and methodological breaks.
- Alcohol denominator, units, recall period, and whether values are
  observed, estimated, modelled, or projected.

Produce a feasibility table before analysis.
Choose the longest defensible overlap with stable measurement,
not the longest period regardless of comparability.

2. CAREER CONCENTRATION

Prefer mutually exclusive first-choice workplace categories.
Label the measure "workplace-aspiration concentration" if the source
measures employer/workplace type rather than occupations.

Let p[k,t] be the weighted share choosing category k in year t.

Compute:
HHI[t] = sum_k p[k,t]**2
C[t] = (HHI[t] - 1/K) / (1 - 1/K)
EffectiveCategories[t] = 1 / HHI[t]

Use a fixed set of K harmonized categories across all years.
Keep zero-share categories.
Document every category mapping.
Do not merge categories differently in different years.
If definitions cannot be harmonized, split the series or stop.

Exclude missing responses consistently and report their share.
If "other" or "undecided" is substantial, explain its treatment and
provide a sensitivity analysis when possible.

Validate that shares sum to approximately one.
For rounded published shares, disclose any small renormalization.
Do not use incomplete top-three preference tables.
Do not normalize multiple-response percentages into first-choice shares.

Optional:
If justified independently of results, calculate a fixed
"conventional/prestige-track aspiration share".
Keep this concept separate from general concentration.

3. ALCOHOL

Primary simple outcome:
Annual litres of pure alcohol per person aged 15+.

Choose either OECD sales-based consumption or WHO total consumption.
Do not splice these definitions together.

Preferred supplementary outcome:
A documented drinking-prevalence indicator for an age group matching
the career survey as closely as possible.

For prevalence:
B[t] = 100 * sum_i(w[i,t] * drinking_indicator[i,t]) / sum_i(w[i,t])

Preserve the official definition and denominator.
If the denominator includes only drinkers, label it accordingly.
Do not compare youth aspirations with all-adult drinking without
clearly flagging the population mismatch.

4. TIDY DATA

Save:
career_shares.csv:
year, category, share, age_group, source_id

career_concentration.csv:
year, hhi, normalized_hhi, effective_categories, age_group

alcohol.csv:
year, alcohol_value, unit, age_group, measure_definition, source_id

analysis.csv:
Only genuinely observed common years for the primary paired analysis.

sources.csv:
source_id, publisher, dataset, table_or_indicator_id, url,
access_date, observation_years, population, definition, notes

Preserve raw source files separately.
Record any manual extraction, transformations, and category mappings.
Use observation year as the time variable.

5. PLOTS

Create:
A. Two vertically aligned panels sharing a year axis:
   - normalized career concentration, with a 0–1 y-axis;
   - alcohol consumption, with its original units.
   Show all available observations for each series.

B. One overlay showing both lines standardized using:
   z[t] = (x[t] - mean(x)) / sample_standard_deviation(x)
   Calculate means and SDs using the same complete-case overlapping
   observation years for both series.
   State that standardization compares movement, not units.

C. A scatterplot of career concentration versus alcohol consumption,
   labelling each observation by year.

Use points to identify observed values.
Do not fabricate annual career data by interpolation.
Connecting observed waves is allowed only as a visual guide.
Mark methodological breaks and retain missing data as missing.
Prefer the two-panel plot over potentially misleading dual y-axes.

6. ASSOCIATION ANALYSIS

Report:
- Number of paired observations and exact years.
- Pearson correlation and Spearman rank correlation, if feasible.
- A scatterplot and a discussion of shared time trends.

As a sensitivity check, calculate changes between consecutive
observed career-survey years using the same endpoints for alcohol:
delta_C = C[t] - C[previous observed year]
delta_A = A[t] - A[previous observed year]

If intervals vary, disclose this and optionally analyze annualized
changes as a separately labelled sensitivity analysis.

Do not treat years as independent without considering serial
dependence.
Do not add many controls or lag specifications to a short series.
Do not present ordinary correlation p-values as definitive evidence.
Do not select lags or definitions based on the strongest result.
If too few observations exist for meaningful inference, provide
descriptive plots and state this clearly.

Discuss possible alternative explanations:
economic conditions, employment insecurity, population composition,
alcohol policy and prices, workplace practices, and major disruptions.

Do not infer individual behaviour from national aggregates.
Do not infer causation from co-movement.

7. PYTHON DELIVERABLE

Create a runnable script using pandas, numpy, matplotlib, and scipy
where appropriate.

The script must:
- Load documented local CSVs reproducibly.
- Validate categories, units, shares, duplicates, and overlapping years.
- Compute concentration and the analysis table.
- Generate all plots as PNGs.
- Export calculated results to CSV.
- Handle missing years without inventing data.
- Fail with clear messages for incompatible inputs.
- Include dependency and execution instructions.

If automated downloading is unavailable, provide exact official export
instructions and accept downloaded files as inputs.
Never substitute synthetic data for the actual study.
A demonstration dataset is allowed only if separately and prominently
labelled synthetic, and never used for research conclusions.

Finish with:
1. Data availability and comparability findings.
2. Actual observation years and number of paired observations.
3. Metric definitions.
4. Source provenance.
5. Descriptive findings and limitations.
6. The executable Python script and generated outputs.