# CLIMATE MODULE (FROZEN)

This folder is intentionally **not active** in the PRISM Engine.
It is a future project sandbox.
Do not expand, modify, or integrate this module until PRISM core is complete.

No imports should reference this folder.
No start/ scripts should call it.
No fetchers or registries should use climate indicators yet.

Status: FROZEN

he `/climate/` folder contains early scaffolding for a future PRISM expansion involving:

- ERA5 reanalysis data  
- NASA & NOAA climate observation sources  
- Composite climate indicators  
- Climate normalization & temporal transforms  
- Data schemas for multidimensional climate variables  

These files were added automatically but **are NOT active** in the PRISM Engine.  
They are **not tested, not integrated, and not part of the runtime pipeline.**

Until PRISM Core (market + macro engine) is fully complete:

- **Do NOT import from this folder**
- **Do NOT build features on top of this folder**
- **Do NOT expand or modify its structure**
- **Do NOT connect climate indicators to the unified panel loader**
- **Do NOT integrate climate data into geometry engines**

This folder is intentionally isolated to avoid interference with:
- Registry v2
- Unified fetcher architecture (FRED/Tiingo)
- Database schema v2
- Start scripts
- Engine loading pipelines
- HTML workflow runner design

---

## 🧊 **Why It’s Frozen**
We must keep PRISM focused until the foundation is complete:

- Unified indicators  
- Unified data domain  
- Lens engines  
- Geometry orchestrators  
- HTML runner  
- Plugin architecture  
- Workflow engine  
- Consistent DB schema  

After that, climate becomes a **second domain**, which PRISM will be capable of analyzing *as its own system* or in *cross-coherence mode* with macro/financial indicators.

---

## 🔒 **Technical Safeguards**

During freeze:

- Importing `climate` will raise an `ImportError`
- Runtime panel loader will reject climate panels
- Update scripts show a warning if climate is detected
- No fetchers use climate sources
- No PRISM engine reads climate indicators
- Tests ensure climate is not activated accidentally

These safeguards prevent unintentional activation.

---

## 🌱 **Future Activation Plan (When Ready)**

Once PRISM Core is complete:

1. Remove freeze guards  
2. Build dedicated `climate_registry.yaml`  
3. Add climate sources to unified fetcher  
4. Implement climate → panel mapping conventions  
5. Add coherence cross-domain workflows  
6. Create dedicated climate dashboards  
7. Introduce temporal & spatial operators  
8. Expand geometry engine for multi-resolution climate signals  

Climate data can then power:

- Cross-domain coherence (macro ↔ climate)  
- Early-warning climate stress indicators  
- Spatio-temporal harmonic analysis  
- Climate-economic structural breaks  
- Regime-switching across Earth systems  

This is the long-term vision — but **not now.**

---

## 📌 **Summary**

This folder stays **exactly where it is**, but:

- It is **inactive**
- It is **sandbox-only**
- It is **not part of PRISM runtime**
- It is **off-limits for changes** until we intentionally unfreeze it

---

## 🧭 **Next Steps**
Focus remains on:

- Unified runner  
- HTML panel selector  
- Workflow system  
- Plugin architecture  
- Lens normalization  
- Cross-engine orchestration  
- DB schema maturity  
- Full testing + calibration  

When ready, we begin PRISM Phase 2: **Earth-System Integration**.
