# Research: public datasets for the AI-recruiter corpus (résumés, job postings, fit labels, skills taxonomy)

Date: 2026-10-08. Method:
- Live checks against the Hugging Face Hub API (`https://huggingface.co/api/datasets/<id>`), the HF dataset viewer API (`https://datasets-server.huggingface.co/{info,size,parquet}?dataset=<id>`) and the Kaggle public API (`https://www.kaggle.com/api/v1/datasets/view/<owner>/<slug>`).
- Direct `curl` downloads of every recommended file. Row counts, label distributions and PII rates below come from loading those files with pandas in a scratch directory.
- Live requests to the ESCO web-service API and the O*NET 31.0 download area.

Every claim cites its source. Anything not confirmed live is marked **UNVERIFIED**. A licence written on a Hugging Face card is the uploader's *declaration*. Where it disagrees with the upstream source, the upstream licence wins, and this is flagged.

## TL;DR

- **Résumés: `opensporks/resumes`.** It holds 2,484 English résumés with a `Category` label (24 occupations) and is released as **CC0-1.0**. It mirrors the Kaggle dataset `snehaanbhawal/resume-dataset`, which Kaggle's API also lists as "CC0: Public Domain".
  - The résumés are real LiveCareer examples with names and employers already replaced by placeholders: 99.96% contain the literal string "Company Name". Only 0.8% contain an email pattern and 2.3% a phone pattern.
  - Download: one 20 MB parquet file.
  - **Gap:** there is no Data Science category, and software engineering appears only inside `INFORMATION-TECHNOLOGY`, which is mostly IT managers and IT specialists.
- **Jobs: EMSCAD, via `victor/real-or-fake-fake-jobposting-prediction`.** It holds 17,880 postings (17,014 real), is **CC0-1.0**, and Kaggle lists the upstream as "CC0: Public Domain".
  - Fields: `title`, `location`, `department`, `company_profile`, `description`, `requirements`, `benefits`, `employment_type`, `required_experience` (seniority), `required_education`, `industry`, `function`, `fraudulent`.
  - Emails, phones and URLs are already masked as `#EMAIL_…#`, `#PHONE_…#` and `#URL_…#`.
  - The postings date from 2012–2014.
  - The `function` values line up well with the résumé categories.
- **Paired fit labels (evaluation only): `cnamuangtoun/resume-job-description-fit`.** It holds 8,000 pairs (6,241 train and 1,759 test) labelled `No Fit` / `Potential Fit` / `Good Fit`.
  - **No licence is declared** and there is no dataset card. Download it at runtime, use it for evaluation only, and **do not commit it**.
  - The permissive alternative is `AzharAli05/Resume-Screening-Dataset` (MIT, 10,174 rows, `select`/`reject` labels). Its résumés are LLM-generated, so its labels are synthetic.
- **Skills taxonomy:**
  - **ESCO web-service API** works with no key or registration (verified with `search?text=python`). The data falls under the "European Commission reuse notice" (Decision 2011/833/EU, which requires attribution). ESCO's bulk download needs an email form, so use the API and cache the JSON.
  - **O*NET 31.0** is under **CC BY 4.0** and has direct CSV URLs that need no registration. In 31.0, "Technology Skills" has been renamed `software_skills.csv`, and "Skills" has been split into `essential_skills.csv` and `transferable_skills.csv`.
- **Avoid as committed data:**
  - `lukebarousse/data_jobs` (Apache-2.0, but **no description column** and data-only roles).
  - `datastax/linkedin_job_postings` (**404/gone**; the `datastax/linkedin_job_listings` copy has no licence).
  - `xanderios/linkedin-job-postings` (card says MIT, but the upstream Kaggle `arshkon/linkedin-job-postings` is **CC BY-SA 4.0**, and the content is scraped from LinkedIn).
  - `jacob-hugging-face/job-descriptions` (llama2 licence, 853 rows).
  - `ahmedheakl/resume-atlas` (MIT, but its text is lowercased and stripped of punctuation, and it contains **apparent real names, emails and phone numbers**).
- **Proposed sample:** six occupations shared by both sets, taking **~33 résumés and 10 jobs each, for 200 résumés and 60 jobs**. See §6. Use synthetic documents to fill the data-science gap.

---

## 1. Résumé / CV datasets

| HF id | Licence (card) | Upstream licence | Rows | Fields | Label | PII | Text quality | Download | Size |
|---|---|---|---|---|---|---|---|---|---|
| **`opensporks/resumes`** | `cc0-1.0` [^os] | Kaggle `snehaanbhawal/resume-dataset`: "CC0: Public Domain" [^kg1] | 2,484 [^osz] | `ID`, `Resume_str`, `Resume_html`, `Category` | 24 categories, ~100–120 each (BPO 22, AUTOMOBILE 36, AGRICULTURE 63) | Anonymised by source: "Company Name" placeholder in 99.96% of rows; email regex hits 0.8%, phone regex 2.3% (measured) | Raw text with spacing artefacts; median 5.9k chars | `https://huggingface.co/datasets/opensporks/resumes/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` (20.0 MB) or the original CSV `…/resolve/main/Resume/Resume.csv` (56.3 MB) | 20 MB parquet |
| `ahmedheakl/resume-atlas` | `mit` [^atl] | Paper arXiv:2406.18125; GitHub `noran-mohamed/Resume-Classification-Dataset` (licence **UNVERIFIED**) | 13,389 | `Category`, `Text` | 43 categories incl. Data Science 299, Python Developer 248, Java Developer 348, HR 360, Sales 364, Finance 339 | **Not anonymised**: sample rows contain full names, emails ("padesmond038gmailcom") and phone numbers; phone regex hits 30% | Lowercased, punctuation and stop-words removed, some OCR noise. Poor for display or RAG | `https://huggingface.co/datasets/ahmedheakl/resume-atlas/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 23.6 MB |
| `InferencePrince555/Resume-Dataset` | `apache-2.0` (card has no other text) [^ip] | Unknown; text looks like LiveCareer résumés with punctuation stripped (**UNVERIFIED** provenance) | 32,481 | `instruction` ("Generate a Resume for a Accountant Job"), `input`, `Resume_test` | Category only inside the `instruction` string | Not measured | Punctuation stripped | `https://huggingface.co/datasets/InferencePrince555/Resume-Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 103 MB |
| `C0ldSmi1e/resume-dataset` | `apache-2.0` | Same LiveCareer source as opensporks (identical category set). Relicensing CC0 as Apache is harmless | 2,257 train + 200 test | `id`, `category`, `skills`, `education`, `experience`, `text`, `html` | 23 categories | Same as opensporks (0.8% email, 2.3% phone) | Same text plus pre-split sections | `https://huggingface.co/datasets/C0ldSmi1e/resume-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 18.9 MB |
| `AzharAli05/Resume-Screening-Dataset` | `mit` | n/a; **LLM-generated** ("Here's a professional resume for Patricia Gray: …") | 10,174 | `Role`, `Resume`, `Decision`, `Reason_for_decision`, `Job_Description` | 45 roles (Data Scientist 538, Software Engineer 480, HR Specialist 262, UX Designer 227 …; casing inconsistent) | Fake contact details (`…@email.com`, `555-555-5555`) in ~89% of rows | Clean but synthetic | `https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 12.7 MB |
| `snehaanbhawal/resume-dataset` (HF) | — | — | — | — | — | — | — | HF API returns "Invalid username or password" (this id does not exist on HF; it is Kaggle-only) | — |

Notes
- The Kaggle original can also be fetched without authentication. `https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset` returned a 302 to a signed GCS zip during this check. This is not documented as stable, so prefer the HF mirror.
- The opensporks repo also contains the original PDFs (`data/data/<CATEGORY>/<ID>.pdf`). These are useful if the project wants a "PDF ingestion" demo.
- Within opensporks, the first token run of `Resume_str` is the résumé headline. For example, ACCOUNTANT résumés are mostly "ACCOUNTANT", "STAFF ACCOUNTANT" or "SENIOR ACCOUNTANT", and HR résumés are "HR GENERALIST", "HR MANAGER" or "HR COORDINATOR". This headline is a cheap secondary label.

## 2. Job posting datasets

| HF id | Licence (card) | Upstream licence | Rows | Fields (title / company / description / skills / location / seniority) | Notes | Download | Size |
|---|---|---|---|---|---|---|---|
| **`victor/real-or-fake-fake-jobposting-prediction`** (EMSCAD) | `cc0-1.0` [^em] | Kaggle `shivamb/real-or-fake-fake-jobposting-prediction`: "CC0: Public Domain"; data from University of the Aegean EMSCAD [^kg2] | 17,880 (17,014 `fraudulent==0`) | `title`; company only as `company_profile` text (no name column); `description` (median 1,027 chars); `requirements` (null in 15%), which acts as the skills field; `location` ("US, NY, New York"); `required_experience` (Mid-Senior / Entry / Associate / Director / Internship / Executive / NaN 39%); plus `department`, `salary_range`, `benefits`, `employment_type`, `required_education`, `industry`, `function`, `telecommuting` | PII masked as `#EMAIL_…#`, `#PHONE_…#`, `#URL_…#`. Real-posting `function` counts: IT 1,717, Sales 1,427, Engineering 1,235, Marketing 820, Design 337, HR 196, Accounting/Auditing 183, Finance 157, Data Analyst 78. Locations: US 9.9k, GB 2.4k, GR 0.9k. Dated 2012–2014 | `https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv` (50.1 MB) or parquet `…/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` (23.6 MB) | 50 MB CSV |
| `gplsi/fake_job_postings_balanced_en` | `cc-by-4.0` | Same EMSCAD (card says original is CC0) | 1,732 (50% fake) | Same as EMSCAD | Balanced *toward fraud*, so half the rows are unusable. 230 duplicate title+description pairs | `https://huggingface.co/datasets/gplsi/fake_job_postings_balanced_en/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 2.1 MB |
| `lukebarousse/data_jobs` | `apache-2.0` [^lb] | Scraped via SerpApi/Google Jobs (upstream terms **UNVERIFIED**) | 785,741 | `job_title_short`, `job_title`, `job_location`, `job_via`, `job_schedule_type`, `company_name`, `job_skills`, `job_type_skills`, salary fields … **no description** | Data roles only (10 classes). Good skill lists, but nothing to embed | `https://huggingface.co/datasets/lukebarousse/data_jobs/resolve/main/data_jobs.csv` | 231 MB CSV / 75 MB parquet |
| `datastax/linkedin_job_postings` | — | — | — | — | **Gone**: HF API returns "Invalid username or password" (repo missing or private) | — | — |
| `datastax/linkedin_job_listings` | none declared | Kaggle `arshkon/linkedin-job-postings`: **CC BY-SA 4.0** [^kg3] | 123,849 | `company_name`, `title`, `description`, `location`, `formatted_experience_level`, `skills_desc`, salary … | ShareAlike plus scraped LinkedIn content (real company names and URLs) | `…/datastax/linkedin_job_listings/resolve/main/postings.csv` | 526 MB |
| `xanderios/linkedin-job-postings` | `mit` (**conflicts with upstream CC BY-SA 4.0**) | Kaggle arshkon (older 33k snapshot, 2023) | 33,246 | `title`, `description`, `location`, `formatted_experience_level`, `skills_desc` (99% null) … no company name | Top titles are sales, accounting and admin; only 50 "Software Engineer" | `https://huggingface.co/datasets/xanderios/linkedin-job-postings/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 62.8 MB |
| `jacob-hugging-face/job-descriptions` | `llama2` [^jc] | Unknown | 853 | `company_name`, `job_description`, `position_title`, `description_length`, `model_response` (Llama-2 JSON extraction) | Small and broad, but llama2 licence and unknown provenance of the descriptions | `https://huggingface.co/datasets/jacob-hugging-face/job-descriptions/resolve/main/training_data.csv` | 3.8 MB |
| `lang-uk/recruitment-dataset-job-descriptions-english` | `mit` | Djinni (Ukrainian IT job board) | 141,897 | `Position`, `Long Description`, `Company Name`, `Exp Years`, `Primary Keyword` … | IT-heavy. Weak for HR, sales and finance | `…/resolve/main/data/train-00000-of-00001.parquet` | 267 MB in memory |
| `makenomistakesllc/tech-job-postings-9-ats` | `cc-by-4.0` | Self-scraped from 9 ATS (2026-09-22) | 7,516 | `company`, `title`, `department`, `city`, `country`, `description_text`, `employment_type`, salary … | Recent and permissive, but real company names and tech-only. Not deeply checked | `…/resolve/main/tech_jobs_9_ats_2026-09-22.parquet` | 43 MB |

## 3. Paired résumé ↔ job datasets with fit labels

| HF id | Licence | Rows | Label scheme | Composition | Download | Size |
|---|---|---|---|---|---|---|
| **`cnamuangtoun/resume-job-description-fit`** | **None declared; no README** ("Entry not found") [^cn] | 6,241 train + 1,759 test | `label` ∈ {`No Fit`, `Potential Fit`, `Good Fit`}. Train: 3,143 / 1,556 / 1,542. Test: 857 / 444 / 458 | `resume_text`, `job_description_text`, `label`. Train has 642 unique résumés × 280 unique JDs. No JD overlap between train and test. Résumés look like the same LiveCareer source; JDs are real staffing-agency posts | train `https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` (9.8 MB); test `…/default/test/0000.parquet` (2.6 MB) | 12.3 MB parquet |
| `AzharAli05/Resume-Screening-Dataset` | `mit` | 10,174 | `Decision` ∈ {`select` 5,060, `reject` 5,114} plus `Reason_for_decision` | LLM-generated résumés with 3,446 unique JDs | see §1 | 12.7 MB |
| `batuhanmtl/job_resume_fit` | `other` (flag) | 2,385 | Continuous `ai_match_score` (0–100, mean 44), `skill_string_match_score`, `fuzzy_match_score`. Scores are AI-generated, not human labels | Kaggle LiveCareer résumés (card says so) × one job text per category; `job_required_skills`, `resume_skill_list` | `https://huggingface.co/datasets/batuhanmtl/job_resume_fit/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet` | 8.2 MB |

How the fit labels in the cnamuangtoun set were produced (human or model) is **UNVERIFIED**, because the dataset has no card.

## 4. Skills taxonomies

### 4.1 ESCO (European Skills, Competences, Qualifications and Occupations)

- **Licence:** the EU Open Data Portal lists every ESCO distribution under the **"European Commission reuse notice"** [^eu]. That notice is Commission Decision 2011/833/EU, which allows commercial and non-commercial reuse with acknowledgement of the source, no distortion of meaning and no Commission liability [^dec]. The ESCO API *software* is EUPL-1.2 [^api].
- **Current version:** ESCO v1.2.1, last updated 10/12/2025, per the portal footer (live fetch).
- **Bulk download:** this is **not usable from a script without a human step**. The download page requires selecting a package, accepting a privacy statement and entering an email address to receive a link [^dl].
- **Web-service API (no key, verified live 2026-10-08):**
  - `GET https://ec.europa.eu/esco/api/search?text=python&language=en&type=skill&limit=3` returned HTTP 200 with `total: 2`. Top hit: "Python (computer programming)", `http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d`.
  - `GET https://ec.europa.eu/esco/api/resource/occupation?uri=http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30&language=en` returned "data scientist" with 23 `hasEssentialSkill` and 16 `hasOptionalSkill` links.
  - Occupation URIs for the target set, all checked live via `search?type=occupation`:

    | Occupation | URI suffix |
    |---|---|
    | software developer | `…/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1` |
    | data scientist | `…/258e46f9-0075-4a2e-adae-1ff0477e0f30` |
    | human resources officer | `…/d3e32e5e-7f24-48e3-b939-e4f800eb62fb` |
    | sales account manager | `…/2eac08c2-a81a-46fc-8d75-eb0e0f3e0f6d` |
    | graphic designer | `…/69bcbb0a-8d80-4ecd-b0a4-9adea2a40de2` |
    | accountant | `…/eda0d957-3c3c-4139-b89a-a18bc9e18897` |
    | financial analyst | `…/8d586ee9-0ab1-4155-a3b0-2ca786c8e48c` |

  - No rate-limit headers were returned. The documented rate limit is **UNVERIFIED**.
- **HF mirrors:** `danieldux/ESCO` (licence `cc`, ISCO hierarchy with only 2,413+604 rows) and `ICILS/isco_esco_occupations_taxonomy` (no licence). Neither is a full skills dump, so do not use them.

### 4.2 O*NET 31.0

- **Licence:** **CC BY 4.0** "except as noted", with required credit text: "This page includes information from the O*NET 31.0 Database by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). Used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA." [^on]
- **Direct CSVs** (verified HTTP 200 with no registration, 2026-10-08) [^ondb]:

  | File | Size | Rows | Columns |
  |---|---|---|---|
  | `https://www.onetcenter.org/dl_files/database/db_31_0_csv/software_skills.csv` (was "Technology Skills") | 3.6 MB | 31,821 | `O*NET-SOC Code`, `Title`, `Workplace Example`, `Element ID`, `Element Name`, `Hot Technology`, `In Demand` |
  | `…/db_31_0_csv/essential_skills.csv` (Reading Comprehension, Critical Thinking …) | 2.3 MB | | Importance (`IM`) and level (`LV`) per occupation |
  | `…/db_31_0_csv/transferable_skills.csv` | 6.1 MB | | Same scale structure as essential skills |
  | `…/db_31_0_csv/occupation_data.csv` | 268 KB | 1,016 | `O*NET-SOC Code`, `Title`, `Description` |
  | `…/db_31_0_csv/sample_of_reported_titles.csv` | 601 KB | | Job-title synonyms; useful for mapping résumé headlines to SOC codes |
  | Full archive: `https://www.onetcenter.org/dl_files/database/db_31_0_csv.zip` | 16.2 MB | | |

- **Version naming:** the old file names (`db_30_x_text/Technology%20Skills.txt`) still resolve for 30.1 and 30.2. Under 31.0 they return 404, so pin the version in the downloader.
- **Example:** `15-2051.00 Data Scientists` rows in `software_skills.csv` include Alteryx, Amazon Redshift and AWS SageMaker, each with Hot Technology / In Demand flags.

---

## 5. Licence flags summary

| Dataset | Safe to commit to a public repo? | Why |
|---|---|---|
| `opensporks/resumes` | **Yes** (CC0) | CC0 upstream and on HF. Source is LiveCareer examples; add a NOTICE crediting Kaggle and LiveCareer as good practice |
| EMSCAD (`victor/…`) | **Yes** (CC0) | CC0 upstream. Credit University of the Aegean EMSCAD |
| O*NET 31.0 extracts | **Yes** (CC BY 4.0) | Use the attribution text above and note changes |
| ESCO API extracts | **Yes, with attribution** | EC reuse notice (2011/833/EU) |
| `AzharAli05/Resume-Screening-Dataset` | Yes (MIT) | Synthetic. Fine as an extra paired set |
| `cnamuangtoun/resume-job-description-fit` | **No: download at runtime only** | No licence declared |
| `batuhanmtl/job_resume_fit` | No | `license: other` with no terms |
| `ahmedheakl/resume-atlas` | Avoid | Apparent real PII; upstream licence UNVERIFIED |
| `xanderios/…`, `datastax/linkedin_job_listings` | Avoid | CC BY-SA 4.0 upstream (ShareAlike) and scraped LinkedIn content |
| `jacob-hugging-face/job-descriptions` | Avoid | llama2 licence and unknown provenance |
| `lukebarousse/data_jobs` | Licence fine (Apache-2.0) | Not useful: no descriptions |

## 6. Recommendation

### 6.1 Résumés: `opensporks/resumes`

- **URL:** `https://huggingface.co/datasets/opensporks/resumes/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet`. To pin to a commit, use `https://huggingface.co/datasets/opensporks/resumes/resolve/main/Resume/Resume.csv`.
- **Columns to use:** `ID` (stable id), `Resume_str` (body), `Category` (label). Derive a `headline` from the first whitespace-separated run of `Resume_str`. Skip `Resume_html`.
- **Filter:** `Resume_str` length between 1,500 and 12,000 characters. This leaves roughly 100–117 per target category.

### 6.2 Jobs: EMSCAD via `victor/real-or-fake-fake-jobposting-prediction`

- **URL:** `https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv`.
- **Columns to use:**
  - `job_id`
  - `title`
  - `location`
  - `department`
  - `company_profile`, which becomes the fictional employer blurb. **Replace it with the fictional company's profile when "re-homing" postings.**
  - `description`
  - `requirements`, which serves as the skills text
  - `employment_type`
  - `required_experience`, which serves as seniority
  - `required_education`
  - `industry`
  - `function`, which serves as the occupation label
- **Filter:**
  - `fraudulent == 0`
  - `requirements` not null
  - `len(description) > 400`
  - `location` starting with `US`/`GB`/`CA`/`IE`/`AU`/`NZ`
- **Counts after filtering:** IT+Engineering 1,648 · Sales 1,065 · Design 217 · Finance+Accounting 246 · HR 136 · Data Analyst 54. Another 67 postings have a data scientist or data analyst title.

### 6.3 Paired fit (evaluation only): `cnamuangtoun/resume-job-description-fit`

- **URL:** `https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet`.
- **Columns to use:** `resume_text`, `job_description_text`, `label`.
- **Handling:** download at runtime into a git-ignored cache. Do not commit it, because no licence is declared.
- **Suggested use:** a stratified ~150-pair sample, 50 per label, from the **test** split as matching ground truth.
- **If a committed, permissively licensed paired set is required instead:** use `AzharAli05/Resume-Screening-Dataset` (MIT) with columns `Role`, `Resume`, `Job_Description` and `Decision`, and accept that it is synthetic.

### 6.4 Taxonomy

- **O*NET 31.0 (bulk, committed extract):**
  - `software_skills.csv` and `occupation_data.csv`, plus optionally `essential_skills.csv` (importance `IM` rows) and `sample_of_reported_titles.csv`.
  - Base URL: `https://www.onetcenter.org/dl_files/database/db_31_0_csv/`.
  - Keep only the SOC codes for the six occupations, e.g. 15-1252.00 Software Developers, 15-2051.00 Data Scientists, 13-1071.00 HR Specialists, 41-3091.00 Sales Representatives, 27-1024.00 Graphic Designers, 13-2011.00 Accountants and Auditors, 13-2051.00 Financial Analysts. The exact SOC code list is from memory, **UNVERIFIED** beyond 15-2051.00 (seen live); confirm it against `occupation_data.csv`.
- **ESCO (API, cached JSON):** for each occupation URI in §4.1, fetch `resource/occupation` and then each `hasEssentialSkill` / `hasOptionalSkill` title. That is about 7 occupations × ~40 skills, which is a few hundred requests once. Commit the resulting JSON with EC attribution.
- **Fit with the project:** ESCO fits the EU framing of the wider project. O*NET supplies concrete tool and technology names such as "Python" and "Tableau" that ESCO's skill labels often phrase abstractly.

### 6.5 Occupations and sample sizes (200 résumés / 60 jobs)

| Occupation | Résumé `Category` | Résumés | EMSCAD filter | Jobs |
|---|---|---|---|---|
| Software / IT | `INFORMATION-TECHNOLOGY` | 34 | `function ∈ {Information Technology, Engineering}` and `title ~ /software\|developer\|engineer\|devops\|web/i` | 10 |
| Human resources | `HR` | 33 | `function == Human Resources` | 10 |
| Sales | `SALES` | 33 | `function == Sales` | 10 |
| Design | `DESIGNER` | 33 | `function == Design` (graphic, UX, product designer) | 10 |
| Finance | `FINANCE` | 33 | `function == Finance` | 10 |
| Accounting | `ACCOUNTANT` | 34 | `function == Accounting/Auditing` | 10 |
| **Total** | | **200** | | **60** |

- **Data science gap:** opensporks has **no data-science category**. Only 34 résumés mention data science, machine learning or data analysis at all, and they are scattered across categories. EMSCAD has only 54–67 data postings.
- **If data science must be one of the six,** generate the ~33 data-science résumés synthetically, which is already part of the corpus plan. Take the 10 data-science jobs from EMSCAD (`function == Data Analyst` or `title ~ /data scien|data analy/i`). Then merge Finance and Accounting into one occupation to stay at six.
- **Sampling:** use a fixed seed, sample per stratum, and spread EMSCAD `required_experience` across Entry, Associate and Mid-Senior so seniority matching has signal.
- **Software résumés:** IT résumés skew toward IT managers and specialists rather than software engineers, so either accept this and frame the role as "IT / software", or top up with synthetic software-engineer résumés.

---

## Sources

[^os]: Hugging Face dataset card and API: https://huggingface.co/datasets/opensporks/resumes · https://huggingface.co/api/datasets/opensporks/resumes (card: `license: cc0-1.0`, `kaggle_id: snehaanbhawal/resume-dataset`; "Resume Examples taken from livecareer.com").
[^osz]: https://datasets-server.huggingface.co/size?dataset=opensporks/resumes (num_rows 2484).
[^kg1]: https://www.kaggle.com/api/v1/datasets/view/snehaanbhawal/resume-dataset (`licenseNameNullable: "CC0: Public Domain"`).
[^atl]: https://huggingface.co/datasets/ahmedheakl/resume-atlas (card `license: mit`; points to https://github.com/noran-mohamed/Resume-Classification-Dataset and https://arxiv.org/abs/2406.18125).
[^ip]: https://huggingface.co/datasets/InferencePrince555/Resume-Dataset (card only `license: apache-2.0`).
[^em]: https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction (card `license: cc0-1.0`; file `fake_job_postings.csv`, X-Linked-Size 50,061,601).
[^kg2]: https://www.kaggle.com/api/v1/datasets/view/shivamb/real-or-fake-fake-jobposting-prediction ("CC0: Public Domain"; acknowledges University of the Aegean, http://emscad.samos.aegean.gr/).
[^lb]: https://huggingface.co/datasets/lukebarousse/data_jobs (card `license: apache-2.0`, data dictionary).
[^kg3]: https://www.kaggle.com/api/v1/datasets/view/arshkon/linkedin-job-postings (`licenseNameNullable: "CC BY-SA 4.0"`).
[^jc]: https://huggingface.co/datasets/jacob-hugging-face/job-descriptions (card `license: llama2`).
[^cn]: https://huggingface.co/api/datasets/cnamuangtoun/resume-job-description-fit (no `license` in cardData; `README.md` → "Entry not found").
[^eu]: https://data.europa.eu/api/hub/search/search?q=ESCO%20skills%20competences%20qualifications%20occupations&filter=dataset → dataset `european-skills-competences-qualifications-and-occupations`, distributions licensed "European Commission reuse notice".
[^dec]: Commission Decision 2011/833/EU on the reuse of Commission documents: https://eur-lex.europa.eu/eli/dec/2011/833/oj/eng.
[^api]: https://esco.ec.europa.eu/en/use-esco/use-esco-services-api ("The ESCO API is licensed under the EUPL 1.2").
[^dl]: https://esco.ec.europa.eu/en/use-esco/download ("Read and accept the statement, then enter your email address. You will receive a link to download your chosen dataset."; versions up to v1.2.1).
[^on]: https://www.onetcenter.org/license_db.html ("O*NET® 31.0 Database Content License … Creative Commons Attribution 4.0 International License").
[^ondb]: https://www.onetcenter.org/database.html (links to `dl_files/database/db_31_0_csv/*.csv` and `db_31_0_csv.zip`).
